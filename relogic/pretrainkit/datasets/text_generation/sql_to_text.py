import logging
import os
from tqdm import tqdm
import json

from dataclasses import dataclass
from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from relogic.pretrainkit.datasets.utils import pad_and_tensorize_sequence

from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

class SQL2QueryDataset(Dataset):
  """
  Dataset for training task: SQL (+ schema) -> text
  """
  def __init__(self, tokenizer: PreTrainedTokenizer, file_path, block_size, local_rank=-1):
    assert os.path.isfile(file_path)
    logger.info("Creating features from dataset file at {}".format(file_path))

    self.examples = []
    total, valid = 0, 0
    add_prefix_space = isinstance(tokenizer, BartTokenizer) or isinstance(tokenizer, RobertaTokenizer)
    with open(file_path, encoding="utf-8") as f:
      for line in tqdm(f):
        total += 1
        example = json.loads(line)
        # sql = example["text"]
        # text = example["labels"]
        sql = " ".join(example["entities"])
        text = example["labels"].strip()

        text_tokens = [tokenizer.cls_token] + tokenizer.tokenize(text, add_prefix_space=add_prefix_space) + [tokenizer.sep_token]
        sql_tokens = [tokenizer.cls_token] + tokenizer.tokenize(sql, add_prefix_space=add_prefix_space) + [tokenizer.sep_token]

        text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        sql_token_ids = tokenizer.convert_tokens_to_ids(sql_tokens)
        self.examples.append({
          "text_token_ids": text_token_ids,
          "sql_token_ids": sql_token_ids})
    logger.info("Total {} examples.".format(total))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    return self.examples[i]

@dataclass
class DataCollatorForSQL2Query:
  """

  """
  tokenizer: PreTrainedTokenizer

  def __post_init__(self):
    self.label_bos_id = self.tokenizer.cls_token_id
    self.label_eos_id = self.tokenizer.sep_token_id

  def collate_batch(self, examples):
    text_ids_sequences = [example["text_token_ids"] for example in examples]
    sql_ids_sequences = [example["sql_token_ids"] for example in examples]

    padded_text_ids_tensor = pad_and_tensorize_sequence(
      text_ids_sequences, padding_value=self.tokenizer.pad_token_id)

    padded_sql_ids_tensor = pad_and_tensorize_sequence(
      sql_ids_sequences, padding_value=self.tokenizer.pad_token_id)

    return {
      "input_ids": padded_sql_ids_tensor,
      "labels": padded_text_ids_tensor,
      "pad_token_id": self.tokenizer.pad_token_id,
      "label_eos_id": self.label_eos_id,
      "label_bos_id": self.label_bos_id,
      "label_padding_id": self.tokenizer.pad_token_id
    }