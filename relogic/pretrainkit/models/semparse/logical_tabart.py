import torch
from transformers import BartForConditionalGeneration
from relogic.pretrainkit.models.semparse.semparse import BartForTextToSQL
import logging
import os
from relogic.logickit.modules.span_extractors.average_span_extractor import AverageSpanExtractor
import json
KEYWORDS =  json.load(open("data/preprocessed_data/bart_parser_label_mapping.json"))["keyword"]

import torch.nn.functional as F
import torch.nn as nn
import functools

def rgetattr(obj, attr, *args):
  def _getattr(obj, attr):
    return getattr(obj, attr, *args)

  return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
  pre, _, post = attr.rpartition(".")
  return setattr(rgetattr(obj, pre) if pre else obj, post, val)


logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"

class LogicalTaBARTModel(nn.Module):
  """
  output: tuple: (loss, ) in training
  """
  def __init__(self, task):
    super().__init__()
    tasks = task.replace(",", "+").split("+")
    self.bert_for_texttosql = BartForTextToSQL.from_pretrained("facebook/bart-large")
    self.bert = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    # self.bert_for_texttosql.model.decoder.layers.__delitem__(slice(6, 12, 1))
    # self.bert.model.decoder.layers.__delitem__(slice(6, 12, 1))
    for name in self.bert_for_texttosql.state_dict().keys():
      if name != 'model.keyword_embedding.weight' and not any(["model.decoder" in name]):
        rsetattr(self.bert, name, rgetattr(self.bert_for_texttosql, name))
    self.average_span_extractor = AverageSpanExtractor()
    self.column_mlp = nn.Linear(self.bert.config.d_model, self.bert.config.d_model)
    if "col_type" in tasks:
      self.column_to_prob = nn.Linear(self.bert.config.d_model, 3)
    else:
      self.column_to_prob = nn.Linear(self.bert.config.d_model, 1)
    self.value_column_mlp = nn.Linear(self.bert.config.d_model * 2, self.bert.config.d_model)
    self.value_column_to_prob = nn.Linear(self.bert.config.d_model, 1)
    self.table_pred_mlp = nn.Linear(self.bert.config.d_model, 2)

  def column_prediction(self, input_ids, attention_mask, column_spans):
    column_mask = (column_spans[:, :, 0] > 0).long()
    features = self.bert.model.encoder(input_ids=input_ids,
                                 attention_mask=attention_mask)[0].contiguous()
    column_features = self.average_span_extractor(
      sequence_tensor=features,
      span_indices=column_spans,
      span_indices_mask=column_mask)
    column_selection_logits = self.column_to_prob(torch.relu(self.column_mlp(column_features)))
    column_selection_prob = torch.sigmoid(column_selection_logits)
    return column_selection_prob
    # return column_selection_logits

  def column_classification(self, input_ids, attention_mask, column_spans):
    column_mask = (column_spans[:, :, 0] > 0).long()
    features = self.bert.model.encoder(input_ids=input_ids,
                                 attention_mask=attention_mask)[0].contiguous()
    column_features = self.average_span_extractor(
      sequence_tensor=features,
      span_indices=column_spans,
      span_indices_mask=column_mask)
    column_selection_logits = self.column_to_prob(torch.relu(self.column_mlp(column_features)))
    return column_selection_logits

  def value_prediction(self, input_ids, attention_mask, column_spans, value_spans):
    column_mask = (column_spans[:, :, 0] > 0).long()
    features = self.bert.model.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask)[0].contiguous()
    column_features = self.average_span_extractor(
      sequence_tensor=features,
      span_indices=column_spans,
      span_indices_mask=column_mask)
    # (batch, k, dim)
    value_feature = self.average_span_extractor(
      sequence_tensor=features,
      span_indices=value_spans)
    # (batch, 1, dim)
    column_features = torch.cat([column_features, value_feature.expand(column_features.size())], dim=-1)
    column_selection_logits = self.value_column_to_prob(torch.relu(self.value_column_mlp(column_features)))
    column_selection_prob = torch.sigmoid(column_selection_logits)
    return column_selection_prob

  def table_pred(self, input_ids, attention_mask):
    features = self.bert.model.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask)[0].contiguous()
    logits = self.table_pred_mlp(features[:, 0])
    return logits


  def forward(self, *input, **kwargs):
    input_ids = kwargs.pop("input_ids")
    pad_token_id = kwargs.pop("pad_token_id")
    attention_mask = (input_ids != pad_token_id).long()


    if self.training:
      task = kwargs.pop("task")
      if task == "text2sql":
        copy_span = None
        column_spans = kwargs.pop("column_spans")
        label_ids = kwargs.pop("labels")
        label_padding_id = kwargs.pop("label_padding_id")
        # encoded = self.bert.encoder(input_token_ids)[0].contiguous()
        y_ids = label_ids[:, :-1].contiguous()
        lm_labels = label_ids[:, 1:].clone()
        lm_labels[label_ids[:, 1:] == label_padding_id] = -100
        outputs = self.bert_for_texttosql(input_ids, column_spans=column_spans, copy_span=copy_span,
                            attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels, )
        return (outputs[0],)

      if task == "mlm" or task == "col_rev":
        output_ids = kwargs.pop('labels')
        y_ids = output_ids[:, :-1].contiguous()
        lm_labels = output_ids[:, 1:].clone()
        lm_labels[output_ids[:, 1:] == pad_token_id] = -100
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels, )
        return (outputs[0],)

      if task == "recurring_mlm":
        y_ids = kwargs.pop("y_ids")
        output_ids = kwargs.pop('labels')
        lm_labels = output_ids[:, 1:].clone()
        lm_labels[output_ids[:, 1:] == pad_token_id] = -100
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels, )
        return (outputs[0],)

      if task == "col_pred":
        label_ids = kwargs.pop("labels")
        column_spans = kwargs.pop("column_spans")
        column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)
        label_mask = column_spans.view(-1, 2)[:,0] > 0

        column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask], label_ids.view(-1)[label_mask].float(),
                                                       reduction="sum") / label_ids.size(0)
        # column_selection_loss = F.cross_entropy(column_selection_prob.view(-1, 3)[label_mask],
        #                                         label_ids.view(-1)[label_mask],
        #                                         reduction="sum") / label_ids.size(0)
        return (column_selection_loss, )

      if task == "col_type":
        label_ids = kwargs.pop("labels")
        column_spans = kwargs.pop("column_spans")
        column_selection_prob = self.column_classification(input_ids, attention_mask, column_spans)
        label_mask = column_spans.view(-1, 2)[:,0] > 0

        column_selection_loss = F.cross_entropy(column_selection_prob.view(-1, 3)[label_mask],
                                                label_ids.view(-1)[label_mask],
                                                reduction="sum") / label_ids.size(0)
        return (column_selection_loss, )

      if task == "value_pred":
        label_ids = kwargs.pop("labels")
        column_spans = kwargs.pop("column_spans")
        value_spans = kwargs.pop("value_spans")
        column_selection_prob = self.value_prediction(input_ids, attention_mask, column_spans, value_spans)
        label_mask = column_spans.view(-1, 2)[:, 0] > 0

        column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask],
                                                       label_ids.view(-1)[label_mask].float(),
                                                       reduction="sum") / label_ids.size(0)
        return (column_selection_loss,)

      if task == "table_pred":
        label_ids = kwargs.pop("labels")
        table_prediction_prob = self.table_pred(input_ids, attention_mask)

        table_prediction_loss = F.cross_entropy(table_prediction_prob.view(-1, 2),
                                                         label_ids.view(-1),
                                                       reduction="sum") / label_ids.size(0)
        return (table_prediction_loss,)


      raise NotImplementedError("Unknown task {}".format(task))

    else:
      task = kwargs.pop("task")
      if task == "text2sql":
        copy_span = None
        column_spans = kwargs.pop("column_spans")
        label_eos_id = kwargs.pop("label_eos_id")
        label_bos_id = kwargs.pop("label_bos_id")
        label_padding_id = kwargs.pop("label_padding_id")
        generated_ids = self.bert_for_texttosql.generate(
          input_ids=input_ids,
          column_spans=column_spans,
          copy_span=copy_span,
          attention_mask=attention_mask,
          num_beams=1,
          max_length=30,
          length_penalty=2.0,
          early_stopping=True,
          use_cache=True,
          decoder_start_token_id=label_bos_id,
          eos_token_id=label_eos_id,
          pad_token_id=label_padding_id,
          vocab_size=len(KEYWORDS)
        )

        output_ids = kwargs.pop("labels")
        y_ids = output_ids[:, :-1].contiguous()
        lm_labels = output_ids[:, 1:].clone()
        lm_labels[output_ids[:, 1:] == label_padding_id] = -100
        outputs = self.bert_for_texttosql(input_ids, column_spans=column_spans,
                            attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels, )

        return (outputs[0].detach(), generated_ids)

      if task == "recurring_mlm":
        label_eos_id = kwargs.pop("label_eos_id")
        label_bos_id = kwargs.pop("label_bos_id")
        label_padding_id = kwargs.pop("label_padding_id")
        generated_ids = self.bert.generate(
          input_ids=input_ids,
          attention_mask=attention_mask,
          num_beams=3,
          max_length=input_ids.size(1) + 5,
          length_penalty=2.0,
          early_stopping=True,
          use_cache=True,
          decoder_start_token_id=label_bos_id,
          eos_token_id=label_eos_id,
          pad_token_id=label_padding_id
        )
        generated_ids = generated_ids[:, 1:].contiguous()
        y_ids = kwargs.pop("y_ids")
        output_ids = kwargs.pop('labels')
        lm_labels = output_ids[:, 1:].clone()
        lm_labels[output_ids[:, 1:] == pad_token_id] = -100
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels, )

        return (outputs[0].detach(), generated_ids)

      if task == "mlm" or task == "col_rev":
        label_eos_id = kwargs.pop("label_eos_id")
        label_bos_id = kwargs.pop("label_bos_id")
        label_padding_id = kwargs.pop("label_padding_id")
        generated_ids = self.bert.generate(
          input_ids=input_ids,
          attention_mask=attention_mask,
          num_beams=3,
          max_length=input_ids.size(1) + 5,
          length_penalty=2.0,
          early_stopping=True,
          use_cache=True,
          decoder_start_token_id=label_bos_id,
          eos_token_id=label_eos_id,
          pad_token_id=label_padding_id
        )
        generated_ids = generated_ids[:,1:].contiguous()
        output_ids = kwargs.pop('labels')
        y_ids = output_ids[:, :-1].contiguous()
        lm_labels = output_ids[:, 1:].clone()
        lm_labels[output_ids[:, 1:] == label_padding_id] = -100

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels, )
        return (outputs[0].detach(), generated_ids)

      if task == "col_pred":
        label_ids = kwargs.pop("labels")
        column_spans = kwargs.pop("column_spans")
        column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)

        generated_ids = (column_selection_prob.squeeze(-1) > 0.5).long()
        generated_ids[column_spans[:,:,0]==0] = -100

        label_mask = column_spans.view(-1, 2)[:, 0] > 0

        column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask],
                                                       label_ids.view(-1)[label_mask].float(),
                                                       reduction="sum") / label_ids.size(0)
        # column_selection_loss = F.cross_entropy(column_selection_prob.view(-1, 3)[label_mask],
        #                                         label_ids.view(-1)[label_mask],
        #                                         reduction="sum") / label_ids.size(0)
        return (column_selection_loss.detach(), generated_ids)

      if task == "col_type":
        label_ids = kwargs.pop("labels")
        column_spans = kwargs.pop("column_spans")
        column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)

        generated_ids = column_selection_prob.argmax(dim=-1)
        generated_ids[column_spans[:, :, 0] == 0] = -100

        label_mask = column_spans.view(-1, 2)[:, 0] > 0
        column_selection_loss = F.cross_entropy(column_selection_prob.view(-1, 3)[label_mask],
                                                label_ids.view(-1)[label_mask],
                                                reduction="sum") / label_ids.size(0)
        return (column_selection_loss.detach(), generated_ids)

      if task == "value_pred":
        label_ids = kwargs.pop("labels")
        column_spans = kwargs.pop("column_spans")
        value_spans = kwargs.pop("value_spans")
        column_selection_prob = self.value_prediction(input_ids, attention_mask, column_spans, value_spans)

        generated_ids = (column_selection_prob.squeeze(-1) > 0.5).long()
        generated_ids[column_spans[:, :, 0] == 0] = -100

        label_mask = column_spans.view(-1, 2)[:, 0] > 0

        column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask],
                                                       label_ids.view(-1)[label_mask].float(),
                                                       reduction="sum") / label_ids.size(0)
        return (column_selection_loss.detach(), generated_ids)

      if task == "table_pred":
        label_ids = kwargs.pop("labels")
        table_prediction_prob = self.table_pred(input_ids, attention_mask)
        generated_ids = table_prediction_prob.argmax(dim=-1).unsqueeze(-1)
        table_prediction_loss = F.cross_entropy(table_prediction_prob.view(-1, 2),
                                                       label_ids.view(-1),
                                                       reduction="sum") / label_ids.size(0)
        return (table_prediction_loss.detach(), generated_ids)

      raise NotImplementedError()

  def save_pretrained(self, save_directory):
    """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
    """
    assert os.path.isdir(
      save_directory
    ), "Saving path should be a directory where the model and configuration can be saved"

    # Only save the model itself if we are using distributed training
    model_to_save = self.module if hasattr(self, "module") else self

    # Attach architecture to the config
    # model_to_save.config.architectures = [model_to_save.__class__.__name__]

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)

    logger.info("Model weights saved in {}".format(output_model_file))



