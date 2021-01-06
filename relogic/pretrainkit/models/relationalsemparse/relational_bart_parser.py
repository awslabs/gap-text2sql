import torch
import torch.nn as nn

from relogic.pretrainkit.models.relationalsemparse.relational_semparse import RelationalBartForTextToSQL
from relogic.logickit.dataflow.semtransparse.grammar.keywords import SKETCH_KEYWORDS, KEYWORDS
import logging
import os

logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"

class RelationalBARTParser(nn.Module):
  """
  output: tuple: (loss, ) in training
  """
  def __init__(self):
    super().__init__()
    self.bert: RelationalBartForTextToSQL = RelationalBartForTextToSQL.from_pretrained("facebook/bart-large")
    self.bert.model.encoder.use_relation_transformer = True


  def forward(self, *input, **kwargs):

    input_token_ids = kwargs.pop("input_ids")
    # column_spans = kwargs.pop("column_spans")
    input_padding_id = kwargs.pop("input_padding_id")
    attention_mask = (input_token_ids != input_padding_id).long()
    example_info_list= kwargs.pop("example_info_list")
    # relation_ids = None
    if self.training:
      label_ids = kwargs.pop("labels")
      label_padding_id = kwargs.pop("label_padding_id")
      # encoded = self.bert.encoder(input_token_ids)[0].contiguous()
      y_ids = label_ids[:, :-1].contiguous()
      lm_labels = label_ids[:, 1:].clone()
      lm_labels[label_ids[:, 1:] == label_padding_id] = -100
      outputs = self.bert(input_token_ids, example_info_list=example_info_list,
                          attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels, )
      return (outputs[0],)

    else:
      label_eos_id = kwargs.pop("label_eos_id")
      label_bos_id = kwargs.pop("label_bos_id")
      label_padding_id = kwargs.pop("label_padding_id")
      generated_ids = self.bert.generate(
        input_ids=input_token_ids,
        example_info_list=example_info_list,
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
      # raise NotImplementedError()
      # label_ids = kwargs.pop("label_ids")
      # label_padding_id = kwargs.pop("label_padding_id")
      # # encoded = self.bert.encoder(input_token_ids)[0].contiguous()
      # y_ids = label_ids[:, :-1].contiguous()
      # lm_labels = label_ids[:, 1:].clone()
      # lm_labels[label_ids[:, 1:] == label_padding_id] = -100
      # outputs = self.bert(input_token_ids, example_info_list=example_info_list,
      #                     attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels, )
      # generated_ids = outputs[-1]
      # raise NotImplementedError()
      return (torch.zeros(1), generated_ids)

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



