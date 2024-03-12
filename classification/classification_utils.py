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
    print(torch.cuda.is_available())
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
