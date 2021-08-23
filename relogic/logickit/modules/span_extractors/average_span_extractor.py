import torch.nn as nn
import torch
from relogic.logickit.utils import utils



class AverageSpanExtractor(nn.Module):
  def __init__(self):
    super(AverageSpanExtractor, self).__init__()

  def forward(self,
              sequence_tensor: torch.FloatTensor,
              span_indices: torch.LongTensor,
              sequence_mask: torch.LongTensor = None,
              span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
    # Shape (batch_size, num_spans, 1)
    span_starts, span_ends = span_indices.split(1, dim=-1)

    span_ends = span_ends - 1

    span_widths = span_ends - span_starts

    max_batch_span_width = span_widths.max().item() + 1

    # sequence_tensor (batch, length, dim)
    # global_attention_logits = self._global_attention(sequence_tensor)
    global_average_logits = torch.ones(sequence_tensor.size()[:2] + (1,)).float().to(sequence_tensor.device)

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = utils.get_range_vector(max_batch_span_width,
                                                    sequence_tensor.device).view(1, 1, -1)
    span_mask = (max_span_range_indices <= span_widths).float()

    # (batch_size, num_spans, 1) - (1, 1, max_batch_span_width)
    raw_span_indices = span_ends - max_span_range_indices
    span_mask = span_mask * (raw_span_indices >= 0).float()
    span_indices = torch.relu(raw_span_indices.float()).long()

    flat_span_indices = utils.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

    span_embeddings = utils.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

    span_attention_logits = utils.batched_index_select(global_average_logits,
                                                       span_indices,
                                                       flat_span_indices).squeeze(-1)

    span_attention_weights = utils.masked_softmax(span_attention_logits, span_mask)

    attended_text_embeddings = utils.weighted_sum(span_embeddings, span_attention_weights)

    if span_indices_mask is not None:
      return attended_text_embeddings * span_indices_mask.unsqueeze(-1).float()

    return attended_text_embeddings





