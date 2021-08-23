import logging
import math
import os
import random
import re
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import itertools

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from transformers.data.data_collator import DataCollator, default_data_collator
from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput

from relogic.pretrainkit.training_args import TrainingArguments
from relogic.pretrainkit.trainer_utils import EvalPredictionWithSize, PredictionOutputWithSize



if is_apex_available():
    from apex import amp


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


def is_wandb_available():
    return False

if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collators: List[DataCollator]
    train_datasets: List[Optional[Dataset]]
    eval_datasets: List[Optional[Dataset]]
    compute_metrics: List[Optional[Callable[[EvalPrediction], Dict]]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collators: List[Optional[DataCollator]] = None,
        train_datasets: List[Optional[Dataset]] = None,
        eval_datasets: List[Optional[Dataset]] = None,
        compute_metrics: List[Optional[Callable[[EvalPrediction], Dict]]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        self.data_collators = data_collators if data_collators is not None else [default_data_collator]
        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets
        if not isinstance(compute_metrics, list):
            self.compute_metrics = [compute_metrics]
        else:
            self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True


    def get_train_dataloaders(self) -> List[DataLoader]:
        if self.train_datasets is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_samplers = [get_tpu_sampler(train_dataset) for train_dataset in self.train_datasets]
        else:
            train_samplers = [(
                RandomSampler(train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(train_dataset)
            ) for train_dataset in self.train_datasets]

        data_loaders = [DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
        ) for dataset, train_sampler, data_collator, train_batch_size in
            zip(self.train_datasets, train_samplers, self.data_collators, self.args.train_batch_sizes)]

        return data_loaders

    def get_eval_dataloaders(self, eval_datasets: List[Optional[Dataset]] = None) -> List[DataLoader]:
        if eval_datasets is None and self.eval_datasets is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_datasets = eval_datasets if eval_datasets is not None else self.eval_datasets

        if is_torch_tpu_available():
            samplers = [SequentialDistributedSampler(
                dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            ) for dataset in eval_datasets]
        elif self.args.local_rank != -1:
            samplers = [SequentialDistributedSampler(dataset) for dataset in eval_datasets]
        else:
            samplers = [SequentialSampler(dataset) for dataset in eval_datasets]

        data_loaders = [DataLoader(
            dataset,
            sampler=sampler,
            batch_size=eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
        ) for dataset, sampler, data_collator, eval_batch_size in
            zip(eval_datasets, samplers, self.data_collators, self.args.eval_batch_sizes)]

        return data_loaders

    def get_test_dataloader(self, test_datasets: List[Dataset]) -> List[DataLoader]:
        # We use the same batch_size as for eval.
        if is_torch_tpu_available():
            samplers = [SequentialDistributedSampler(
                dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            ) for dataset in test_datasets]
        elif self.args.local_rank != -1:
            samplers = [SequentialDistributedSampler(dataset) for dataset in test_datasets]
        else:
            samplers = [SequentialSampler(dataset) for dataset in test_datasets]

        data_loader = [DataLoader(
            dataset,
            sampler=sampler,
            batch_size=eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
        ) for dataset, sampler, data_collator, eval_batch_size in
            zip(test_datasets, samplers, self.data_collators, self.args.eval_batch_sizes)]

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        if self.is_world_master():
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                wandb.watch(
                    self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
                )

    def num_examples(self, dataloaders: Union[DataLoader, List[DataLoader]]) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        if isinstance(dataloaders, DataLoader):
            return len(dataloaders.dataset)
        else:
            return sum([len(dataloader.dataset) for dataloader in dataloaders])

    def get_training_mb(self, dataloaders: List[DataLoader], epoch):
        stop_indices = [False] * len(dataloaders)
        iterators = [iter(dataloader) for dataloader in dataloaders]
        train_steps = [math.ceil(len(dataloader.dataset) / dataloader.batch_size) for dataloader in dataloaders]
        switch = list(itertools.chain.from_iterable([[idx] * step for idx, step in enumerate(train_steps)]))
        g = torch.Generator()
        g.manual_seed(epoch)
        indices = torch.randperm(len(switch), generator=g).tolist()
        for idx in indices:
            batch = next(iterators[switch[idx]], None)
            if batch is not None:
                yield batch
            else:
                stop_indices[switch[idx]] = True
            if all(stop_indices):
                break
        return

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloaders = self.get_train_dataloaders()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (sum([len(train_dataloader) for train_dataloader in train_dataloaders]) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(sum([len(train_dataloader) for train_dataloader in train_dataloaders]) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = [
                train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            for train_batch_size in self.args.train_batch_sizes]
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloaders))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %s", str(self.args.per_device_train_batch_sizes))
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %s", str(total_train_batch_size))
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (sum([len(train_dataloader) for train_dataloader in train_dataloaders]) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    sum([len(train_dataloader) for train_dataloader in train_dataloaders]) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master() or not self.args.logging_tqdm
        )

        for epoch in train_iterator:
            for train_dataloader in train_dataloaders:
                if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loaders = [pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                ) for train_dataloader in train_dataloaders]
                epoch_iterators = parallel_loaders
            else:
                epoch_iterators = train_dataloaders

            epoch_iterator_length = sum([len(epoch_iterator) for epoch_iterator in epoch_iterators])

            for step, inputs in enumerate(self.get_training_mb(epoch_iterators, epoch)):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    epoch_iterator_length <= self.args.gradient_accumulation_steps
                    and (step + 1) == epoch_iterator_length
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / epoch_iterator_length

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        learning_rates = scheduler.get_last_lr() if version.parse(torch.__version__) >= version.parse("1.4") else scheduler.get_lr()[0]
                        for group_idx, learning_rate in enumerate(learning_rates):
                            logs["learning_rate_{}".format(group_idx)] = learning_rate
                        logging_loss = tr_loss

                        self._log(logs)

                    if (self.args.eval_steps > 0 and self.global_step % self.args.eval_steps == 0):
                        if self.args.evaluate_during_training:
                            self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    for epoch_iterator in epoch_iterators:
                        epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, None)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        if is_wandb_available():
            if self.is_world_master():
                wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(output)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def is_local_master(self) -> bool:
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # if not isinstance(self.model, PreTrainedModel):
        #     raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self, eval_datasets: List[Optional[Dataset]] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloaders = self.get_eval_dataloaders(eval_datasets)
        metrics = {}
        for idx, eval_dataloader in enumerate(eval_dataloaders):
            output = self._prediction_loop(idx, eval_dataloader, description="Evaluation")

            self._log(output.metrics)
            metrics.update(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return metrics

    def predict(self, test_datasets: List[Dataset]) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        outputs = []
        for idx, test_dataset in enumerate(test_datasets):
            test_dataloader = self.get_test_dataloader(test_dataset)
            outputs.append(self._prediction_loop(idx, test_dataloader, description="Prediction"))
        return outputs

    def _prediction_loop(
        self, idx: int, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.

        NOTE: One issue is on the size of prediction and labels.
        For current code, it considers all the prediction and labels in different batch have same length of sequence.
        This is not true for our application. To make this more general, I will reformat the predictions and labels.

        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        preds_size: torch.Tensor = None
        label_ids: torch.Tensor = None
        label_size: torch.Tensor = None
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels",
                                                                 "typing_prediction_index", "label_prediction_index"])

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                # Change the way of concat
                # We need to make sure that the size of preds and labels is (batch_size, sequence_length)
                if preds is None:
                    preds = logits.detach()
                    preds_size = preds.new_full(size=preds.size()[:1], fill_value=preds.size(1)).detach()
                    preds = preds.view(-1)
                else:
                    preds_size = torch.cat((preds_size, logits.new_full(size=logits.size()[:1], fill_value=logits.size(1)).detach()), dim=0)
                    preds = torch.cat((preds, logits.detach().view(-1)), dim=0)

                # label_name = None
                # if inputs.get("labels") is not None:
                #     label_name = "labels"
                # if inputs.get("lm_labels") is not None:
                #     label_name = "lm_labels"
                # if inputs.get("masked_lm_labels") is not None:
                #     label_name = "lm_labels"
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                        label_size = label_ids.new_full(size=label_ids.size()[:1], fill_value=label_ids.size(1)).detach()
                        label_ids = label_ids.view(-1)
                    else:
                        label_size = torch.cat((label_size, inputs["labels"].new_full(size=inputs["labels"].size()[:1], fill_value=inputs["labels"].size(1)).detach()), dim=0)
                        label_ids = torch.cat((label_ids, inputs["labels"].detach().view(-1)), dim=0)
                # if inputs.get("label_prediction_index") is not None and inputs.get(
                #     "typing_prediction_index") is not None:
                #     pass


        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                # preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
                preds, preds_size = self.distributed_concat_with_size(preds, preds_size, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                # label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
                label_ids, label_size = self.distributed_concat_with_size(label_ids, label_size, num_total_examples=self.num_examples(dataloader))
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            # NOTE: We do not modify this for now.
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
            preds_size = preds_size.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()
            label_size = label_size.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            # metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            metrics = self.compute_metrics[idx](EvalPredictionWithSize(predictions=preds, predictions_size=preds_size, label_ids=label_ids, label_size=label_size))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        # return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
        return PredictionOutputWithSize(predictions=preds, predictions_size=preds_size, label_ids=label_ids, label_size=label_size, metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output

    def distributed_concat_tensor(self, tensor: torch.Tensor):
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)
        return concat

    def distributed_concat_varsize_tensor(self, tensor: torch.Tensor):
        assert self.args.local_rank != -1

        sizes = self.distributed_concat_tensor(tensor.new_full(size=(1,), fill_value=tensor.size(0)))
        max_size = sizes.max().item()

        padded = tensor.new_zeros(max_size)
        padded[:tensor.size(0)] = tensor

        padded_agg = self.distributed_concat_tensor(padded)
        slices = []
        for i, size in enumerate(sizes):
            start_idx = i * max_size
            end_idx = start_idx + size.item()
            slices.append(padded_agg[start_idx: end_idx])
        ret = torch.cat(slices, dim=0)
        return ret


    def distributed_concat_with_size(self, tensor: torch.Tensor, size: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        # output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        # output_sizes = [size.clone() for _ in range(torch.distributed.get_world_size())]
        # torch.distributed.all_gather(output_tensors, tensor)
        # torch.distributed.all_gather(output_sizes, size)
        # concat = torch.cat(output_tensors, dim=0)
        # concat_sizes = torch.cat(output_sizes, dim=0)
        concat_sizes = self.distributed_concat_varsize_tensor(size)
        concat = self.distributed_concat_varsize_tensor(tensor)

        # output_sizes = concat_sizes[:num_total_examples]

        assert concat_sizes.sum() == concat.size(0)
        return concat, concat_sizes



