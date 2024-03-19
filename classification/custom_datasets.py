import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Lambda


class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.features = X
        # self.features = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(y.to_numpy())  # , dtype=torch.float32
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features.iloc[idx]['review']
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        """
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        """
        return {'input_ids': feature[0], 'attention_mask': feature[1], 'label': label}


def one_hot_encode(num_labels: int):
    return Lambda(lambda y: torch.zeros(num_labels, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))


class CustomBaselineDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, num_labels: int, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.target_transform = target_transform
        # self.target_transform = target_transform if target_transform is not None else one_hot_encode(num_labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        review = row["review"]
        label = row["is-review-bombing"]
        # review = self.dataframe.iloc[idx, self.dataframe.columns.get_loc("review")]
        # label = self.dataframe.iloc[idx, self.dataframe.columns.get_loc("is-review-bombing")]
        if self.transform:
            review = self.transform(review)
        if self.target_transform:
            label = self.target_transform(label)
            print(label)
        return review, label


class BertCustomDataset(Dataset):
    # Taken from https://wellsr.com/python/fine-tuning-bert-for-sentiment-analysis-with-pytorch/
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['review']
        labels = self.data.iloc[index][['is-review-bombing']].values.astype(int)
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'][0]  # .flatten()
        attention_mask = encoding['attention_mask'][0]
        # resize the tensors to the same size
        input_ids = nn.functional.pad(input_ids, (0, self.max_length - input_ids.shape[0]), value=0)
        attention_mask = nn.functional.pad(attention_mask, (0, self.max_length - attention_mask.shape[0]), value=0)
        return input_ids, attention_mask, torch.tensor(labels)
