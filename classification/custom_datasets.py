import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    # noinspection PyPep8Naming
    def __init__(self, X, y, transform=None, target_transform=None):
        self.features = X
        # self.features = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(y.to_numpy())  # , dtype=torch.float32
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        feature = self.features.iloc[idx]['review']
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)

        # torch.tensor(label, dtype=torch.long)
        return {'input_ids': feature[0], 'attention_mask': feature[1], 'labels': label}
        """
        input_ids = self.features.iloc[idx]['input_ids']
        attention_mask = self.features.iloc[idx]['attention_mask']
        label = self.labels[idx]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}


class CustomBaselineDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.target_transform = target_transform

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
