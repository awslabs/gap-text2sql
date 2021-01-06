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
import random

logger = logging.getLogger(__name__)

class MultiDDataset(Dataset):
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

        sql = " ".join(example["sql"].split()).lower()
        text = example["question"].strip().lower()

        text_tokens = [tokenizer.cls_token] + tokenizer.tokenize(text, add_prefix_space=add_prefix_space) + [tokenizer.sep_token]
        sql_tokens = [tokenizer.cls_token] + tokenizer.tokenize(sql, add_prefix_space=add_prefix_space) + [tokenizer.sep_token]

        text_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        sql_token_ids = tokenizer.convert_tokens_to_ids(sql_tokens)
        if len(text_token_ids) > 800 or len(sql_token_ids) > 800:
          continue

        self.examples.append({
          "text_token_ids": text_token_ids,
          "sql_token_ids": sql_token_ids})
    logger.info("Total {} examples.".format(total))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    return self.examples[i]

@dataclass
class DataCollatorForMultiD:
  """

  """
  tokenizer: PreTrainedTokenizer
  bi_direc: bool = False

  def __post_init__(self):
    # self.nl_token_id = self.tokenizer.convert_tokens_to_ids(["<nl>"])[0]
    # self.sql_token_id = self.tokenizer.convert_tokens_to_ids(["<sql>"])[0]
    # self.label_bos_id = [self.nl_token_id, self.sql_token_id]# self.tokenizer.cls_token_id
    self.label_eos_id = self.tokenizer.sep_token_id
    self.label_bos_id = self.tokenizer.cls_token_id


  def collate_batch(self, examples):
    text_ids_sequences = [example["text_token_ids"] for example in examples]
    sql_ids_sequences = [example["sql_token_ids"] for example in examples]

    padded_text_ids_tensor = pad_and_tensorize_sequence(
      text_ids_sequences, padding_value=self.tokenizer.pad_token_id)

    padded_sql_ids_tensor = pad_and_tensorize_sequence(
      sql_ids_sequences, padding_value=self.tokenizer.pad_token_id)

    if self.bi_direc:
      if random.random() < 0.5:
        return {
          "input_ids": padded_sql_ids_tensor,
          "labels": padded_text_ids_tensor,
          "pad_token_id": self.tokenizer.pad_token_id,
          "label_eos_id": self.label_eos_id,
          "label_bos_id": self.label_bos_id,
          "label_padding_id": self.tokenizer.pad_token_id
        }

      else:
        return {
          "input_ids": padded_text_ids_tensor,
          "labels": padded_sql_ids_tensor,
          "pad_token_id": self.tokenizer.pad_token_id,
          "label_eos_id": self.label_eos_id,
          "label_bos_id": self.label_bos_id,
          "label_padding_id": self.tokenizer.pad_token_id
        }
    else:
      return {
        "input_ids": padded_text_ids_tensor,
        "labels": padded_sql_ids_tensor,
        "pad_token_id": self.tokenizer.pad_token_id,
        "label_eos_id": self.label_eos_id,
        "label_bos_id": self.label_bos_id,
        "label_padding_id": self.tokenizer.pad_token_id
      }