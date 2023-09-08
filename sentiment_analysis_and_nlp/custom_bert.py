# Code taken from https://github.com/1tangerine1day/Aspect-Term-Extraction-and-Analysis

from transformers import BertModel
from torch.utils.data import Dataset
import torch


class dataset_ATM(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values

        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')
        pols = pols.strip('][').split(', ')

        bert_tokens = []
        bert_tags = []
        bert_pols = []
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            bert_tokens += t
            bert_tags += [int(tags[i])] * len(t)
            bert_pols += [int(pols[i])] * len(t)

        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        tags_tensor = torch.tensor(bert_tags)
        pols_tensor = torch.tensor(bert_pols)

        return bert_tokens, ids_tensor, tags_tensor, pols_tensor

    def __len__(self):
        return len(self.df)


class dataset_ABSA(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values
        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')
        pols = pols.strip('][').split(', ')

        bert_tokens = []
        bert_att = []
        pols_label = 0
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            bert_tokens += t
            if int(pols[i]) != -1:
                bert_att += t
                pols_label = int(pols[i])

        segment_tensor = [0] + [0] * len(bert_tokens) + [0] + [1] * len(bert_att)
        bert_tokens = ['[cls]'] + bert_tokens + ['[sep]'] + bert_att

        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        pols_tensor = torch.tensor(pols_label)
        segment_tensor = torch.tensor(segment_tensor)

        return bert_tokens, ids_tensor, segment_tensor, pols_tensor

    def __len__(self):
        return len(self.df)


class bert_ATE(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(bert_ATE, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, tags_tensors, masks_tensors):
        bert_outputs, _ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors)
        # print(bert_outputs.size())
        linear_outputs = self.linear(bert_outputs)
        # print(linear_outputs.size())

        if tags_tensors is not None:
            tags_tensors = tags_tensors.view(-1)
            linear_outputs = linear_outputs.view(-1, 3)
            # print(linear_outputs.size())
            # print(tags_tensors.size())
            loss = self.loss_fn(linear_outputs, tags_tensors)
            return loss
        else:
            return linear_outputs


class bert_ABSA(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(bert_ABSA, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, lable_tensors, masks_tensors, segments_tensors):
        _, pooled_outputs = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors)
        # print(bert_outputs.size())
        linear_outputs = self.linear(pooled_outputs)
        # print(linear_outputs.size())

        if lable_tensors is not None:
            # print(linear_outputs.size())
            # print(tags_tensors.size())
            loss = self.loss_fn(linear_outputs, lable_tensors)
            return loss
        else:
            return linear_outputs
