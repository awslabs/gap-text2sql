import attr
import math
import torch
import transformers

from seq2struct.utils import registry

registry.register('optimizer', 'adadelta')(torch.optim.Adadelta)
registry.register('optimizer', 'adam')(torch.optim.Adam)
registry.register('optimizer', 'sgd')(torch.optim.SGD)


@registry.register('lr_scheduler', 'warmup_polynomial')
@attr.s
class WarmupPolynomialLRScheduler:
    param_groups = attr.ib()
    num_warmup_steps = attr.ib()
    start_lr = attr.ib()
    end_lr = attr.ib()
    decay_steps = attr.ib()
    power = attr.ib()

    def update_lr(self, current_step):
        if current_step < self.num_warmup_steps:
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        else:
            new_lr = (
                (self.start_lr - self.end_lr) * (1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                + self.end_lr)

        for param_group in self.param_groups:
            param_group['lr'] = new_lr


@registry.register('lr_scheduler', 'warmup_polynomial_group')
@attr.s
class WarmupPolynomialLRSchedulerGroup(WarmupPolynomialLRScheduler):
    start_lrs = attr.ib()
    """
    Each param group has it's own start lr
    start lr is in the same order as param groups,
    """

    def update_lr(self, current_step):
        for start_lr, param_group in zip(self.start_lrs, self.param_groups):
            if current_step < self.num_warmup_steps:
                warmup_frac_done = current_step / self.num_warmup_steps
                new_lr = start_lr * warmup_frac_done
            else:
                new_lr = (
                    (start_lr - self.end_lr) * (1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                    + self.end_lr)

            param_group['lr'] = new_lr


@registry.register('lr_scheduler', 'warmup_cosine')
@attr.s
class WarmupCosineLRScheduler:
    param_groups = attr.ib()
    num_warmup_steps = attr.ib()
    start_lr = attr.ib()
    end_lr = attr.ib()
    decay_steps = attr.ib()

    def update_lr(self, current_step):
        if current_step < self.num_warmup_steps:
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        else:
            new_lr = (
                (self.start_lr - self.end_lr) * 0.5 * (1 + math.cos(math.pi * (current_step - self.num_warmup_steps) / self.decay_steps))
                + self.end_lr)

        for param_group in self.param_groups:
            param_group['lr'] = new_lr


@registry.register('lr_scheduler', 'noop')
class NoOpLRScheduler:
    def __init__(self, optimizer):
        pass

    def update_lr(self, current_step):
        pass

@registry.register('optimizer', 'bertAdamw')
class BertAdamW(transformers.AdamW):
    """
    Given a model and its bert module, create parameter groups with different lr
    """
    def __init__(self, non_bert_params, bert_params, lr=1e-3, bert_lr=2e-5, **kwargs):
        self.bert_param_group = {"params" : bert_params , "lr": bert_lr, "weight_decay": 0}
        self.non_bert_param_group = {"params" : non_bert_params} 

        params = [self.non_bert_param_group, self.bert_param_group]
        if "name" in kwargs: del kwargs["name"] #TODO: fix this
        super(BertAdamW, self).__init__(params, lr=lr, **kwargs)

@registry.register('lr_scheduler', 'bert_warmup_polynomial_group')
@attr.s
class BertWarmupPolynomialLRSchedulerGroup(WarmupPolynomialLRScheduler):
    """
    Set the lr of bert to be zero when the other param group is warming-up
    """
    start_lrs = attr.ib()

    # Bert parameters are in the second group by default
    def update_lr(self, current_step):
        for i, (start_lr, param_group) in enumerate(zip(self.start_lrs, self.param_groups)):
            if current_step < self.num_warmup_steps:
                if i == 0:
                    warmup_frac_done = current_step / self.num_warmup_steps
                    new_lr = start_lr * warmup_frac_done
                else: # fix bert during warm-up
                    assert i == 1
                    new_lr = 0
            else:
                new_lr = (
                    (start_lr - self.end_lr) * (1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                    + self.end_lr)

            param_group['lr'] = new_lr


@registry.register('optimizer', 'adamw')
class AdamW(torch.optim.Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    **Modified to implement AdamW, see https://arxiv.org/pdf/1711.05101v3.pdf**
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # MODIFIED HERE
                #if group['weight_decay'] != 0:
                #    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # MODIFIED HERE
                # Note that weight_decay is ultimately multiplied with the learning rate.
                update = exp_avg / denom
                if group['weight_decay'] != 0:
                    update += group['weight_decay'] * p.data
                p.data.add_(-step_size * update)

                #p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
