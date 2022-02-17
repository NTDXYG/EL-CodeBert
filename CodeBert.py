import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

class TextRNNAtten(nn.Module):
    def __init__(self, config):
        super(TextRNNAtten, self).__init__()
        hidden_size = config.hidden_size #隐藏层数量
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels ##类别数
        num_layers = 2 ##双层LSTM

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=classifier_dropout)
        # self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x = [batch size, 12, hidden_size]
        x = self.dropout(x)
        # x = [batch size, 12, hidden_size]
        output, (hidden, cell) = self.lstm(x)
        # output = [batch size, 12, num_directions * hidden_size]
        M = self.tanh(output)
        # M = [batch size, 12, num_directions * hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # alpha = [batch size, 12, 1]
        out = output * alpha
        # print(alpha)
        # out = [batch size, 12, num_directions * hidden_size]
        out = torch.sum(out, 1)
        # out = [batch size, num_directions * hidden_size]
        out = F.gelu(out)
        # out = [batch size, num_directions * hidden_size]
        out = self.dense(out)
        # out = [batch size, hidden_size]
        out = self.dropout(out)
        # out = [batch size, hidden_size]
        out = self.fc(out)
        # out = [batch size, num_classes]
        return out, alpha

class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        hidden_size = config.hidden_size #隐藏层数量
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels ##类别数
        num_layers = 2 ##双层LSTM

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=classifier_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # [batch size, 12, hidden_size]
        x = self.dropout(x)
        # [batch size, text size, hidden_size]
        output, (hidden, cell) = self.lstm(x)
        # output = [batch size, text size, num_directions * hidden_size]
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.fc(output[:, -1, :])  # 句子最后时刻的 hidden state
        # output = [batch size, num_classes]
        return output

class Atten(nn.Module):
    def __init__(self, config):
        super(Atten, self).__init__()
        hidden_size = config.hidden_size #隐藏层数量
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels ##类别数

        self.fc = nn.Linear(hidden_size, num_classes)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        # x = [batch size, 12, hidden_size]
        x = self.dropout(x)
        # x = [batch size, 12, hidden_size]
        M = self.tanh(x)
        # M = [batch size, 12, hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # alpha = [batch size, 12, 1]
        out = x * alpha
        # out = [batch size, 12, hidden_size]
        out = torch.sum(out, 1)
        # out = [batch size, hidden_size]
        # out = F.gelu(out)
        # out = [batch size, hidden_size]
        out = self.fc(out)
        # out = [batch size, num_classes]
        return out

class EL_CodeBert_wo_LSTM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier_lstm = Atten(config)
        self.init_weights()
        for p in self.roberta.parameters():
            p.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作Bi-LSTM的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        logits = self.classifier_lstm(cls_embeddings)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class EL_CodeBert_wo_Attention(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier_lstm = TextRNN(config)
        self.init_weights()
        for p in self.roberta.parameters():
            p.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作Bi-LSTM的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        logits = self.classifier_lstm(cls_embeddings)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class EL_CodeBert(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier_lstm = TextRNNAtten(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作Bi-LSTM的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        logits, alpha = self.classifier_lstm(cls_embeddings)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=alpha,
        )