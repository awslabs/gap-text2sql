import numpy as np
import torch
from typing import Optional
from relogic.logickit.base import utils
import itertools

def print_2d_tensor(tensor):
  """ Print a 2D tensor """
  utils.log("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
  for row in range(len(tensor)):
    if tensor.dtype != torch.long:
      utils.log(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
    else:
      utils.log(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))

def entropy(p):
  """Compute the entropy of a probability distribution"""
  plogp = p * torch.log(p)
  plogp[p == 0] = 0
  return -plogp.sum(dim=-1)

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=-1)

def sigmoid(x):
  """Compute sigmoid values"""
  return np.exp(-np.logaddexp(0, -x))

def gen_position_indicator(span, length):
  indicator = [0] * length
  for idx, i in enumerate(range(span[0], -1, -1)):
    indicator[i] = -idx
  for idx, i in enumerate(range(span[1], length)):
    indicator[i] = idx + 1
  return indicator

def indicator_vector(index, length, default_label=0, indicator_label=1, head_index=None):
  vector = [default_label] * length
  if head_index is None:
    for idx in index:
      vector[idx] = indicator_label
  else:
    for idx in index:
      vector[head_index[idx]] = indicator_label
  return vector

def truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def get_span_labels(sentence_tags, is_head=None, segment_id=None, inv_label_mapping=None, ignore_label=list([])):
  """Go from token-level labels to list of entities (start, end, class)."""
  if inv_label_mapping:
    sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
  filtered_sentence_tag = []
  if is_head:
    # assert(len(sentence_tags) == len(is_head))

    for idx, (head, segment) in enumerate(zip(is_head, segment_id)):
      if (head == 1 or head == True) and (segment == 0 or segment == True):
        if sentence_tags[idx] != 'X':
          filtered_sentence_tag.append(sentence_tags[idx])
        else:
          filtered_sentence_tag.append("O")
  if filtered_sentence_tag:
    sentence_tags = filtered_sentence_tag
  span_labels = []
  last = 'O'
  start = -1
  for i, tag in enumerate(sentence_tags):
    items = (None, 'O') if tag == 'O' else tag.split('-', 1)
    pos, _ = items if len(items) == 2 else (items[0], None)
    if (pos == 'S' or pos == 'B' or tag == 'O') and last != 'O':
      span_labels.append((start, i - 1, None if len(last.split('-', 1)) != 2 else last.split('-', 1)[-1]))
    if pos == 'B' or pos == 'S' or last == 'O':
      start = i
    last = tag
  if sentence_tags[-1] != 'O':
    span_labels.append((start, len(sentence_tags) - 1,
                        None if len(last.split('-', 1)) != 2 else last.split('-', 1)[-1]))

  # This code has problem!
  # for item in span_labels:
  #   if item[2] in ignore_label:
  #     span_labels.remove(item)

  filtered_labels = []
  for item in span_labels:
    if item[2] not in ignore_label:
      filtered_labels.append(item)

  return set(filtered_labels), sentence_tags

def filter_head_prediction(sentence_tags, is_head):
  filtered_sentence_tag = []
  for idx, head in enumerate(is_head):
    if head == 1 or head == True:
      if sentence_tags[idx] != 'X':
        filtered_sentence_tag.append(sentence_tags[idx])
      else:
        filtered_sentence_tag.append("O")
  return filtered_sentence_tag

def create_tensor(features, attribute, dtype, device):
  try:
    return torch.tensor([getattr(f, attribute) for f in features], dtype=dtype).to(device)
  except Exception as e:
    if attribute not in create_tensor.attribute_warning:
      print("Exception in attribute {}".format(attribute))
      print(e)
      create_tensor.attribute_warning.add(attribute)
    return None

create_tensor.attribute_warning = set([])

def create_tensor_by_stacking(features, attribute, dtype, device):
  try:
    return torch.tensor(
      list(itertools.chain(*[getattr(f, attribute) for f in features])), dtype=dtype).to(device)
  except Exception as e:
    if attribute not in create_tensor.attribute_warning:
      print("Exception in attribute {}".format(attribute))
      print(e)
      create_tensor.attribute_warning.add(attribute)
    return None

create_tensor_by_stacking.attribute_warning = set([])

def pad_sequences(sequences, padding_value):
  max_size = max([len(sequence) for sequence in sequences])
  padded_sequences = []
  for sequence in sequences:
    padded_sequence = sequence + [padding_value] * (max_size - len(sequence))
    padded_sequences.append(padded_sequence)
  return padded_sequences

def get_range_vector(size: int, device) -> torch.Tensor:
  """
  """
  return torch.arange(0, size, dtype=torch.long).to(device)

def flatten_and_batch_shift_indices(indices: torch.LongTensor,
                                    sequence_length: int) -> torch.Tensor:
  """``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor,
  which has size ``(batch_size, sequence_length, embedding_size)``. This function returns a vector
  that correctly indexes into the flattened target. The sequence length of the target must be provided
  to compute the appropriate offset.

  Args:
    indices (torch.LongTensor):

  """
  if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
    raise ValueError("All the elements should be in range (0, {}), but found ({}, {})".format(
      sequence_length - 1, torch.min(indices).item(), torch.max(indices).item()))
  offsets = get_range_vector(indices.size(0), indices.device) * sequence_length
  for _ in range(len(indices.size()) - 1):
    offsets = offsets.unsqueeze(1)

  # (batch_size, d_1, ..., d_n) + (batch_size, 1, ..., 1)
  offset_indices = indices + offsets

  # (batch_size * d_1 * ... * d_n)
  offset_indices = offset_indices.view(-1)
  return offset_indices


def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
  """Select ``target`` of size ``(batch_size, sequence_length, embedding_size)`` with ``indices`` of
  size ``(batch_size, d_1, ***, d_n)``.

  Args:
    target (torch.Tensor): A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).

  """
  if flattened_indices is None:
    flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

  # Shape: (batch_size * sequence_length, embedding_size)
  flattened_target = target.view(-1, target.size(-1))

  # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
  flattened_selected = flattened_target.index_select(0, flattened_indices)
  selected_shape = list(indices.size()) + [target.size(-1)]

  # Shape: (batch_size, d_1, ..., d_n, embedding_size)
  selected_targets = flattened_selected.view(*selected_shape)
  return selected_targets

def batched_index_select_tensor(target: torch.Tensor,
                                indices: torch.LongTensor,
                                flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
  """Select ``target`` of size ``(batch_size, sequence_length, d2, d3, ... ,dn)`` with indices of size
  ``(batch, index_of_sequence)``.
  Args:
    target (torch.Tensor): High dimensional tensor, at least 3-D.
    indices (torch.LongTensor): A 2 dimensional tensor of shape (batch_size, sequence_length)
  """
  if flattened_indices is None:
    flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

  flatten_target = target.view((target.size(0) * target.size(1),) + (target.size()[2:]))

  flattened_selected = flatten_target.index_select(0, flattened_indices)
  selected_shape = indices.size() + target.size()[2:]

  selected_targets = flattened_selected.view(*selected_shape)
  return selected_targets


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
  """
  ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
  masked. This performs a softmax on just the non-masked positions of ``vector``. Passing ``None``
  in for the mask is also acceptable, which is just the regular softmax.

  """
  if mask is None:
    result = torch.softmax(vector, dim=dim)
  else:
    mask = mask.float()
    while mask.dim() < vector.dim():
      mask = mask.unsqueeze(1)
    masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
    result = torch.softmax(masked_vector, dim=dim)
  return result

def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """
  ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
  masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
  ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
  ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
  broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
  unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
  do it yourself before passing the mask into this function.
  In the case that the input vector is completely masked, the return value of this function is
  arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
  of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
  that we deal with this case relies on having single-precision floats; mixing half-precision
  floats with fully-masked vectors will likely give you ``nans``.
  If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
  lower), the way we handle masking here could mess you up.  But if you've got logit values that
  extreme, you've got bigger problems than this.
  """
  if mask is not None:
    mask = mask.float()
    while mask.dim() < vector.dim():
      mask = mask.unsqueeze(1)
    # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
    # results in nans when the whole vector is masked.  We need a very small value instead of a
    # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
    # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
    # becomes 0 - this is just the smallest value we can actually use.
    vector = vector + (mask + 1e-45).log()
  return torch.nn.functional.log_softmax(vector, dim=dim)

def weighted_sum(matrix: torch.Tensor,
                 attention: torch.Tensor) -> torch.Tensor:
  """

  Args:
    matrix ():
    attention ():

  """
  if attention.dim() == 2 and matrix.dim() == 3:
    return attention.unsqueeze(1).bmm(matrix).squeeze(1)
  if attention.dim() == 3 and matrix.dim() == 3:
    return attention.bmm(matrix)
  if matrix.dim() - 1 < attention.dim():
    expanded_size = list(matrix.size())
    for i in range(attention.dim() - matrix.dim() + 1):
      matrix = matrix.unsqueeze(1)
      expanded_size.insert(i + 1, attention.size(i + 1))
    matrix = matrix.expand(*expanded_size)
  intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
  return intermediate.sum(dim=-2)


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
  """
  """
  if tensor.dim() != mask.dim():
    raise ValueError("tensor.dim() {} != mask.dim() {}.".format(tensor.dim(), mask.dim()))
  return tensor.masked_fill((1-mask).byte(), replace_with)

def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
  """Generate mask from variable ``(batch_size,)`` which represents the sequence lengths of each
  batch element.

  Returns:
    torch.Tensor: ``(batch_size, max_length)``
  """
  ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
  range_tensor = ones.cumsum(dim=1)
  return (range_tensor <= sequence_lengths.unsqueeze(1)).long()

def get_device_of(tensor: torch.Tensor) -> int:
  """
  Returns the device of the tensor.
  """
  if not tensor.is_cuda:
    return -1
  else:
    return tensor.get_device()

def batch_sequence_feature_selection(target: torch.Tensor,
                                     indices: torch.LongTensor,
                                     mask: Optional[torch.LongTensor] = None):
  """
  Select ``target`` of size ``(batch_size, sequence_length, embedding_size)`` with indices of size
  ``(batch_size, d)``. The mask is optional of size ``(batch_size, d)``. The return value is of size
  ``(batch_size, d, embedding_size)``
  """
  expanded_indices_shape = (indices.size(0), indices.size(1), target.size(2))
  selected_target = torch.gather(target, 1, indices.unsqueeze(2).expand(expanded_indices_shape))
  if mask is not None:
    selected_target = selected_target * mask.unsqueeze(-1)
  return selected_target

def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
  """
  A numerically stable computation of logsumexp. This is mathematically equivalent to
  `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
  probabilities.

  Parameters
  ----------
  tensor : torch.FloatTensor, required.
      A tensor of arbitrary size.
  dim : int, optional (default = -1)
      The dimension of the tensor to apply the logsumexp to.
  keepdim: bool, optional (default = False)
      Whether to retain a dimension of size one at the dimension we reduce over.
  """
  max_score, _ = tensor.max(dim, keepdim=keepdim)
  if keepdim:
    stable_vec = tensor - max_score
  else:
    stable_vec = tensor - max_score.unsqueeze(dim)
  return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()