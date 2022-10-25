import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.utils import logging

from transformers_mrc import KoElectraConfig
from transformers_mrc import QuestionAnsweringModelOutput
from transformers import ElectraPreTrainedModel, ElectraModel


logger = logging.get_logger(__name__)


KOELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'monologg/koelectra-base-v3-discriminator': "https://huggingface.co/monologg/koelectra-base-v3-discriminator/raw/main/pytorch_model.bin",
    'monologg/koelectra-small-v3-discriminator': "https://huggingface.co/monologg/koelectra-small-v3-discriminator/raw/main/pytorch_model.bin"
}


class ElectraForQuestionAnsweringAVPool(ElectraPreTrainedModel):
    config_class = KoElectraConfig
    base_model_prefix = "koelectra"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        ''' has_ans 추가 '''
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(input_ids, attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids, position_ids=position_ids,
                                                   head_mask=head_mask, inputs_embeds=inputs_embeds,
                                                   output_attentions=output_attentions,
                                                   output_hidden_states=output_hidden_states,)

        sequence_output = discriminator_hidden_states[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        ''' has_log 추가 '''
        has_log = self.has_ans(first_word)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ''' is_impossibles 추가 '''
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            '''choice_loss 추가 & total_loss 계산 식 변경'''
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3

        ''' output 변경 -> has_log 추가됨 '''
        if not return_dict:
            output = (
                start_logits,
                end_logits,
                has_log,
            ) + discriminator_hidden_states[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_log=has_log,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
