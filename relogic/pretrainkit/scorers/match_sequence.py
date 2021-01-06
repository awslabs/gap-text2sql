import json
import torch

def is_rank_0():
  if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
      return True
  else:
    return True
  return False

class MatchSequenceScorer:
  def __init__(self, bos_id, eos_id, output_path):
    if isinstance(bos_id, list):
      self.bos_ids = bos_id
    else:
      self.bos_ids = [bos_id]
    self.eos_id = eos_id
    self.output_path = output_path

  def __call__(self, prediction):
    preds = prediction.predictions
    preds_size = prediction.predictions_size
    label_ids = prediction.label_ids
    label_size = prediction.label_size
    p_start, l_start = 0, 0
    correct, total = 0, 0
    if is_rank_0():
      fout = open(self.output_path, "w")
    for idx, (p_size, l_size) in enumerate(zip(preds_size, label_size)):
      p_end = p_start + p_size
      l_end = l_start + l_size
      pred = self.get_sequence(preds[p_start: p_end])
      label = self.get_sequence(label_ids[l_start: l_end])
      p_start = p_end
      l_start = l_end
      if pred == label:
        correct += 1
      total += 1
      if is_rank_0():
        fout.write(
          json.dumps({
            "idx": idx,
            "pred": pred,
            "label": label}) + "\n")
    return {
      "accuracy": correct / total,
      "correct": correct,
      "total": total
    }


  def get_sequence(self, seq):
    processed_seq = []
    for idx in seq:
      # if idx in self.bos_ids:
      #   continue
      if idx == self.eos_id:
        break
      processed_seq.append(int(idx))
    return processed_seq




