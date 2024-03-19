"""
Utilities specifically for classification, e.g. Transformers, Model Helpers, etc.
"""
import os
import random
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
import torch.backends.cudnn
import torch.backends.cuda
from torch.utils.data import random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def set_random_seed(seed: int = 42, is_pytorch: bool = True) -> None:
    """
    Method taken from article https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    np.random.seed(seed)
    random.seed(seed)
    if is_pytorch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # tensorflow is expected if it is not pytorch
        tf.random.set_seed(seed)
        tf.experimental.numpy.random.seed(seed)
        # When running on the CuDNN backend, two further options must be set
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[INFO] Random seed set as {seed}\n")


def check_system_for_cuda(is_pytorch: bool = True):
    print(torch.cuda.is_available())  # should be True
    print(torch.backends.cuda.is_built())
    print(torch.backends.cudnn.is_available())
    print(torch.backends.cudnn.version())
    print(torch.backends.cudnn.enabled)


def move_tensor_to_gpu(tensor) -> torch.Tensor:
    # We move our tensor to the GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
    return tensor


def get_pytorch_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def df_to_tensor(df: pd.DataFrame):
    """
    Convert a pandas dataframe to a pytorch tensor
    IMPORTANT: the dataframe must contain only numeric values!
    """
    return torch.from_numpy(df.values).to(get_pytorch_device())


def get_vocabularies(df: pd.DataFrame, categorical_columns: List):
    """
    Use the function like this:
        categorical_features = ['uid', 'ugender', 'iid', 'igenre']
        vocab_sizes = get_vocabularies(df, categorical_features)
    """
    vocab_sizes = {}
    for cat in categorical_columns:
        vocab_sizes[cat] = df[cat].max() + 1
    return vocab_sizes


def split_data_scikit(x_data, y_data, test_split=0.2):
    return train_test_split(x_data, y_data, test_size=test_split, stratify=y_data)


def split_data_pandas(data: pd.DataFrame, test_split=0.2):
    # alternative with pandas sample
    train_set = data.sample(frac=1-test_split)
    test_set = data.drop(train_set.index).sample(frac=1.0)
    return train_set, test_set


def split_data_pytorch(data: pd.DataFrame, test_split=0.2):
    # alternative with pytorch random split
    train_data_len = round(len(data) * (1-test_split))  # 80-20 - train-test-split
    test_data_len = len(data) - train_data_len
    train_set, test_set = random_split(data, [train_data_len, test_data_len])
    return train_set, test_set


def encode_target_variable(data: pd.DataFrame, column_names: list[str], use_label_encoder=True):
    if use_label_encoder:
        encoder = LabelEncoder()
        # fit on one column first and use transform afterwards so all "Ja"/"Nein" are encoded the same way in each column
        encoder.fit(data[['is-review-bombing']])
        data[column_names] = data[column_names].apply(encoder.transform)   # .astype('float32')
    else:
        label_mapping = {'Ja': 0, 'Nein': 1}  # uses the same encoding as the label encoder above
        data[column_names] = data[column_names].replace(label_mapping)
        data[column_names] = data[column_names]   # .astype("float32")

    # alternative: one-hot-encoding
    # encoded_df = pd.get_dummies(data, columns=column_names)
