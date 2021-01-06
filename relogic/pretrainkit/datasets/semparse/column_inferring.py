from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm
import json
import random
import itertools
random.seed(3435)
import torch
from dataclasses import dataclass

label_mapping =  json.load(open("data/preprocessed_data/bart_parser_label_mapping_2.json"))

def pad_and_tensorize_sequence(sequences, padding_value):
  max_size = max([len(sequence) for sequence in sequences])
  padded_sequences = []
  for sequence in sequences:
    padded_sequence = sequence + [padding_value] * (max_size - len(sequence))
    padded_sequences.append(padded_sequence)
  return torch.tensor(padded_sequences, dtype=torch.long)

class ColumnInferringDataset(Dataset):
  def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1):
    self.examples = []
    with open(file_path, encoding="utf-8") as f:
      for line in tqdm(f):
        example = json.loads(line)
        # header = example["header"]
        # table_value = example["table"]
        # table_name = example["table_name"]
        # column_type = example["column_type"]
        # caption = example["caption"]
        self.examples.append(example)

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    return self.examples[i]

@dataclass
class DataCollatorForColumnInferring:
  tokenizer: PreTrainedTokenizer
  keywords = label_mapping["keyword"]
  label_padding_id = label_mapping["keyword"].index(label_mapping["label_padding_token"])
  label_eos_id = label_mapping["keyword"].index(label_mapping["label_eos_token"])
  label_bos_id = label_mapping["keyword"].index(label_mapping["label_bos_token"])
  def collate_batch(self, examples):
    training_examples = self.create_example(examples)
    training_tensors = self.create_tensor(training_examples)
    batched_training_examples = self.create_batch(training_tensors)
    return batched_training_examples

  def create_batch(self, examples):
    input_ids_sequences = [example["input_ids"] for example in examples]
    column_spans_sequences = [example["column_spans"] for example in examples]
    label_ids_sequences = [example["output_ids"] for example in examples]
    copy_span = [example["copy_span"] for example in examples]

    padded_input_ids_tensor = pad_and_tensorize_sequence(
      input_ids_sequences, padding_value=self.tokenizer.pad_token_id)
    padded_column_spans_tensor = pad_and_tensorize_sequence(
      column_spans_sequences, padding_value=(0, 1))
    label_ids_tensor = pad_and_tensorize_sequence(
      label_ids_sequences, padding_value=self.label_padding_id)

    return {
      "input_ids": padded_input_ids_tensor,
      "column_spans": padded_column_spans_tensor,
      "copy_span": copy_span,
      "labels": label_ids_tensor,
      "input_padding_id": self.tokenizer.pad_token_id,
      "label_padding_id": self.label_padding_id,
      "label_eos_id": self.label_eos_id,
      "label_bos_id": self.label_bos_id
    }

  def create_tensor(self, examples):
    processed_examples = []
    for example in examples:
      input_seq = example["input"]
      input_words = [self.tokenizer.tokenize(word, add_prefix_space=True) for word in input_seq]
      input_tokens = list(itertools.chain.from_iterable(input_words))
      columns = example["columns"]
      column_start = len(input_tokens) + 2 # add cls and sep
      column_tokens = []
      column_spans = []
      for column in columns:
        column_text = column.replace("_", " ").replace(".", " ")
        tokens = self.tokenizer.tokenize(column_text, add_prefix_space=True)
        column_tokens.extend(tokens)
        column_tokens.append(self.tokenizer.sep_token)
        column_spans.append((column_start, column_start + len(tokens)))
        column_start += len(tokens) + 1

      output_seq = example["output"]
      output_ids = []
      column_count = -1
      # for idx, token in enumerate(output_seq):
      #   if token in columns:
      #     output_ids.append(len(self.keywords) + len(input_tokens) + columns.index(token))
      #     column_count += 1
      #   elif token in self.keywords:
      #     output_ids.append(self.keywords.index(token))
      #   else:
      #     # It should match the column_count-th value in the input_word_ids
      #     base = len(list(itertools.chain.from_iterable(input_words[:column_count]))) + len(self.keywords)
      #     for i in range(len(input_words[column_count])):
      #       output_ids.append(base + i)
      for idx, token in enumerate(output_seq):
        if token in columns:
          output_ids.append(len(self.keywords) + columns.index(token))
          column_count += 1
        elif token in self.keywords:
          output_ids.append(self.keywords.index(token))
        else:
          raise NotImplementedError()
      output_ids = [self.label_bos_id] + output_ids + [self.label_eos_id]
      input_ids = self.tokenizer.convert_tokens_to_ids(
        [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token] + column_tokens)
      if len(input_ids) > 800:
        continue
      processed_examples.append(
        {
          "input_ids": input_ids,
          "output_ids": output_ids,
          "column_spans": column_spans,
          "copy_span": (1, len(input_tokens) + 1),
          "input_tokens": [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token] + column_tokens,
        })

    return processed_examples



  def create_example(self, examples):
    # We will mix several examples together.
    training_examples = []
    batch_start = 0
    while batch_start < len(examples):
      samples = examples[batch_start:batch_start + 2]
      columns = list(itertools.chain.from_iterable([sample["header"] for sample in samples]))
      augmented_columns = list(itertools.chain.from_iterable([sample["augmented_columns"] for sample in samples]))
      column_idx_to_sample_ids = list(
        itertools.chain.from_iterable([[idx] * len(sample["header"]) for idx, sample in enumerate(samples)]))
      column_idx_to_column_seq = list(
        itertools.chain.from_iterable(list(range(len(sample["header"]))) for sample in samples))
      column_size = len(columns)
      selected_columns = random.sample(range(column_size), int(column_size * 0.4))
      batch_start += 3

      processed_example = []
      column_candidates = []
      for column_idx in range(column_size):
        sample_idx = column_idx_to_sample_ids[column_idx]
        table_name = samples[sample_idx]["table_name"]
        column_candidates.append("{}.{}".format(table_name.lower().replace(" ", "_"),
                                                columns[column_idx].lower().replace(" ", "_")))

        if column_idx in selected_columns:

          if random.random() < 0.3:
            column_text = augmented_columns[column_idx].lower()
          else:
            column_text = columns[column_idx].lower()
          processed_example.append(("column", column_text, table_name, columns[column_idx]))
        else:
          column_seq = column_idx_to_column_seq[column_idx]
          row_size = len(samples[sample_idx]["table"])
          value = ""
          try_count = 0
          while not (len(value) > 0 and value != "-"):
            selected_row = random.choice(range(row_size))
            value = samples[sample_idx]["table"][selected_row][column_seq]
            if try_count > 5:
              break
            try_count += 1
          if len(value) > 0 and value != "-":
            processed_example.append(("value", value, table_name, columns[column_idx]))

      random.shuffle(processed_example)
      input_sequence, output_sequence = [], []
      for item in processed_example:
        if item[0] == "column":
          input_sequence.append(item[1])
          output_sequence.append(item[2].lower().replace(" ", "_") + "." + item[3].lower().replace(" ", "_"))
        else:
          input_sequence.append(item[1])
          output_sequence.append(item[2].lower().replace(" ", "_") + "." + item[3].lower().replace(" ", "_"))
          # output_sequence.append("=")
          # output_sequence.append(item[1])
      training_examples.append({
        "input": input_sequence,
        "output": output_sequence,
        "columns": column_candidates
      })
    return training_examples


