import torch
import torch.nn as nn
from transformers.modeling_bart import BartForConditionalGeneration
import logging
import os
from relogic.logickit.modules.span_extractors.average_span_extractor import AverageSpanExtractor
import torch.nn.functional as F

logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"

class TaBARTModel(nn.Module):
  """
  output: tuple: (loss, ) in training
  """
  def __init__(self):
    super().__init__()
    self.bert = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    self.average_span_extractor = AverageSpanExtractor()
    self.column_mlp = nn.Linear(self.bert.config.d_model, self.bert.config.d_model)
    self.column_to_prob = nn.Linear(self.bert.config.d_model, 1)

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


  def forward(self, *input, **kwargs):
    input_ids = kwargs.pop("input_ids")

    pad_token_id = kwargs.pop("pad_token_id")
    attention_mask = (input_ids != pad_token_id).long()

    if self.training:
      task = kwargs.pop("task")
      if task == "mlm":
        output_ids = kwargs.pop('labels')
        y_ids = output_ids[:, :-1].contiguous()
        lm_labels = output_ids[:, 1:].clone()
        lm_labels[output_ids[:, 1:] == pad_token_id] = -100

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels, )
        return (outputs[0],)
      elif task == "col_pred":
        label_ids = kwargs.pop("labels")
        column_spans = kwargs.pop("column_spans")
        column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)
        label_mask = column_spans.view(-1, 2)[:,0] > 0

        column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask], label_ids.view(-1)[label_mask].float(),
                                                       reduction="sum") / label_ids.size(0)
        return (column_selection_loss, )
      else:
        raise NotImplementedError("Unknown task {}".format(task))

    else:
      task = kwargs.pop("task")

      if task == "mlm":
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

        output_ids = kwargs.pop('labels')
        y_ids = output_ids[:, :-1].contiguous()
        lm_labels = output_ids[:, 1:].clone()
        lm_labels[output_ids[:, 1:] == pad_token_id] = -100

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels, )

        return (outputs[0].detach(), generated_ids)
      elif task == "col_pred":
        label_ids = kwargs.pop("labels")
        column_spans = kwargs.pop("column_spans")
        column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)

        generated_ids = (column_selection_prob.squeeze(-1) > 0.5).long()
        generated_ids[column_spans[:,:,0]==0] = -100

        label_mask = column_spans.view(-1, 2)[:, 0] > 0

        column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask],
                                                       label_ids.view(-1)[label_mask].float(),
                                                       reduction="sum") / label_ids.size(0)
        return (column_selection_loss.detach(), generated_ids)


      else:
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



