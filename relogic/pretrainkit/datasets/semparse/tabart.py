from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm
import json
from dataclasses import dataclass
import torch
from relogic.pretrainkit.datasets.utils import pad_and_tensorize_sequence
import random

class TaBARTDataset(Dataset):
  """
  This dataset is used for pretraining task on generation-based or retrieval-based
  text-schema pair examples.
  The fields that will be used is `question`, `table_info.header`, `entities`.
  We already make sure that every entity in `entities` will be in `table_info.header`.
  """
  def __init__(self,
               tokenizer: PreTrainedTokenizer,
               file_path: str,
               col_token: str):
    self.examples = []
    total = 0
    valid = 0
    with open(file_path, encoding="utf-8") as f:
      for line in tqdm(f):
        total += 1
        example = json.loads(line)
        text = example["question"]
        schema = example["table_info"]["header"]
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(text, add_prefix_space=True) + [col_token]
        column_spans = []
        start_idx = len(tokens)
        for column in schema:
          column_tokens = tokenizer.tokenize(column.lower(), add_prefix_space=True)
          tokens.extend(column_tokens)
          column_spans.append((start_idx, start_idx + len(column_tokens)))
          tokens.append(col_token)
          start_idx += len(column_tokens) + 1
        # Change last col token to sep token
        tokens[-1] = tokenizer.sep_token
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        entities = example["entities"]
        column_labels = [0] * len(schema)
        for entity in entities:
          if entity != "limit" and entity != "*":
            column_labels[schema.index(entity)] = 1
        if len(input_ids) > 600:
          continue
        self.examples.append({
          "input_ids": input_ids,
          "column_spans": column_spans,
          "column_labels": column_labels
        })
        valid += 1
        # Create input
    print("Total {} and Valid {}".format(total, valid))
  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    return self.examples[i]


@dataclass
class DataCollatorForTaBART:
  tokenizer: PreTrainedTokenizer
  task: str
  mlm_probability: float = 0.35



  def __post_init__(self):
    self.label_bos_id = self.tokenizer.cls_token_id
    self.label_eos_id = self.tokenizer.sep_token_id

  def collate_batch(self, examples):
    input_ids_sequences = [example["input_ids"] for example in examples]
    padded_input_ids_tensor = pad_and_tensorize_sequence(input_ids_sequences,
                                                         padding_value=self.tokenizer.pad_token_id)
    if self.task == "mlm":
      inputs, labels = self.mask_tokens(padded_input_ids_tensor.clone())
      return {
        "task": "mlm",
        "input_ids": inputs,
        "labels": padded_input_ids_tensor,
        "pad_token_id": self.tokenizer.pad_token_id,
        "label_bos_id": self.tokenizer.bos_token_id,
        "label_eos_id": self.tokenizer.eos_token_id,
        "label_padding_id": self.tokenizer.pad_token_id}
    elif self.task == "col_pred":
      column_labels_sequences = [example["column_labels"] for example in examples]
      padded_label_ids_tensor = pad_and_tensorize_sequence(column_labels_sequences,
                                                           padding_value=-100)
      column_spans_sequences = [example["column_spans"] for example in examples]
      padded_column_spans_tensor = pad_and_tensorize_sequence(column_spans_sequences,
                                                              padding_value=(0, 1))
      return {
        "task": "col_pred",
        "input_ids": padded_input_ids_tensor,
        "column_spans": padded_column_spans_tensor,
        "labels": padded_label_ids_tensor,
        "pad_token_id": self.tokenizer.pad_token_id}
    elif self.task == "mlm+col_pred":
      if random.random() < 0.6:
        inputs, labels = self.mask_tokens(padded_input_ids_tensor.clone())
        return {
          "task": "mlm",
          "input_ids": inputs,
          "labels": padded_input_ids_tensor,
          "pad_token_id": self.tokenizer.pad_token_id,
          "label_bos_id": self.tokenizer.bos_token_id,
          "label_eos_id": self.tokenizer.eos_token_id,
          "label_padding_id": self.tokenizer.pad_token_id}
      else:
        column_labels_sequences = [example["column_labels"] for example in examples]
        padded_label_ids_tensor = pad_and_tensorize_sequence(column_labels_sequences,
                                                             padding_value=-100)
        column_spans_sequences = [example["column_spans"] for example in examples]
        padded_column_spans_tensor = pad_and_tensorize_sequence(column_spans_sequences,
                                                                padding_value=(0, 1))
        return {
          "task": "col_pred",
          "input_ids": padded_input_ids_tensor,
          "column_spans": padded_column_spans_tensor,
          "labels": padded_label_ids_tensor,
          "pad_token_id": self.tokenizer.pad_token_id}

  def mask_tokens(self, inputs):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if self.tokenizer.mask_token is None:
      raise ValueError(
        "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
      )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    special_tokens_mask = [
      self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if self.tokenizer._pad_token is not None:
      padding_mask = labels.eq(self.tokenizer.pad_token_id)
      probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

