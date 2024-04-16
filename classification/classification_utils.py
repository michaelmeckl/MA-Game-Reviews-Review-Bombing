"""
Utilities specifically for classification, e.g. Transformers, Model Helpers, etc.
"""
import itertools
import os
import pathlib
import random
import sys
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
import torch.backends.cudnn
import torch.backends.cuda
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from torch.utils.data import random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from classification.classification_constants import RANDOM_SEED


def set_random_seed(seed: int = RANDOM_SEED, is_pytorch: bool = True) -> None:
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
    if torch.cuda.is_available():
        print('Running on the GPU')
    else:
        print('Running on the CPU')
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
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def save_model_checkpoint(model_object, optimizer_object, epoch, output_path: str | pathlib.Path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_object.state_dict(),
        'optimizer_state_dict': optimizer_object.state_dict(),
        # 'loss': loss,
    }, output_path)


def load_model_checkpoint(model_object, optimizer_object, model_path: str | pathlib.Path):
    # see https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # use model.train() to resume training or model.eval() to start inference
    checkpoint = torch.load(model_path)
    model_object.load_state_dict(checkpoint['model_state_dict'])
    optimizer_object.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    return epoch


def split_data_scikit(x_data, y_data, test_split=0.2):
    return train_test_split(x_data, y_data, test_size=test_split, stratify=y_data)


def split_data_pandas(data: pd.DataFrame, test_split=0.2):
    # alternative with pandas sample
    train_set = data.sample(frac=1 - test_split)
    test_set = data.drop(train_set.index).sample(frac=1.0)
    return train_set, test_set


def split_data_pytorch(data: pd.DataFrame, test_split=0.2):
    # alternative with pytorch random split
    train_data_len = round(len(data) * (1 - test_split))  # 80-20 - train-test-split
    test_data_len = len(data) - train_data_len
    train_set, test_set = random_split(data, [train_data_len, test_data_len])
    return train_set, test_set


def encode_target_variable(data: pd.DataFrame, target_col: str, column_names: list[str], use_label_encoder=True):
    if use_label_encoder:
        encoder = LabelEncoder()
        # fit on one column first and use transform afterwards so all "Ja"/"Nein" are encoded the same way in each column
        encoder.fit(data[[target_col]].values.ravel())
        data[column_names] = data[column_names].apply(encoder.transform)  # .astype('float32')
    else:
        label_mapping = {'Ja': 0, 'Nein': 1}  # use the same encoding as the label encoder above
        data[column_names] = data[column_names].replace(label_mapping)
        data[column_names] = data[column_names]  # .astype("float32")

    # alternative: one-hot-encoding
    # encoded_cols = pd.get_dummies(data, columns=column_names)
    # encoded_df = pd.concat([data, encoded_cols], axis=1).reset_index(drop=True)


def show_training_plot(train_accuracy, val_accuracy, train_loss, val_loss, output_folder=".",
                       output_name="train_history", show=True):
    """
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # save plot to file and show in a new window
    plt.savefig(os.path.join(output_folder, f"{output_name}.svg"), format="svg")
    if show:
        plt.show()
    """

    # use seaborn instead of matplotlib to make it prettier
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (8, 6)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(train_loss, 'b-o', label="Train loss")
    ax1.plot(val_loss, 'g-o', label="Val loss")
    ax2.plot(train_accuracy, 'b-o', label="Train accuracy")
    ax2.plot(val_accuracy, 'g-o', label="Val accuracy")
    fig.suptitle("Training & Validation Results")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax2.set_xlabel("Epoch")
    ax1.set_xticks(np.arange(len(train_loss)))
    ax2.set_xticks(np.arange(len(train_accuracy)))
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    # save plot to file and show in a new window
    plt.savefig(os.path.join(output_folder, f"{output_name}.svg"), format="svg")
    if show:
        fig.show()


def plot_confusion_matrix(cm, class_names, output_folder="."):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Taken from https://www.tensorflow.org/tensorboard/image_summaries#building_an_image_classifier and slightly adjusted

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    # row_sum = cm.sum(axis=1)[:, np.newaxis] if all(cm.sum(axis=1)) != 0 else [1, np.newaxis]
    # labels = np.around((cm.astype('float') / row_sum) if all(cm.sum(axis=1)) != 0 else cm.astype('float'), decimals=2)
    labels = np.around(cm.astype('float'), decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.show()


def calculate_prediction_results(true_labels, predicted_labels, class_names=None):
    label_names = class_names if class_names is not None else ["Ja", "Nein"]

    print(f"\nAccuracy score on test data: {metrics.accuracy_score(true_labels, predicted_labels) * 100:.2f} %")
    print(f"Balanced accuracy score on test data: {metrics.balanced_accuracy_score(true_labels, predicted_labels):.2f}")
    print(f"Precision score on test data: "
          f"{metrics.precision_score(true_labels, predicted_labels, average='weighted'):.2f}")
    print(f"F1 score on test data: {metrics.f1_score(true_labels, predicted_labels, average='weighted'):.2f}")
    try:
        print(f"\nClassification Report:\n"
              f"{metrics.classification_report(true_labels, predicted_labels, target_names=label_names)}")
    except Exception as e:
        sys.stderr.write(f"Failed to compute classification report: {e}")

    # compute and show the confusion matrix
    try:
        conf_matrix = metrics.confusion_matrix(predicted_labels, true_labels, normalize="all")
        print(f"Confusion Matrix:\n{conf_matrix}")
        plot_confusion_matrix(conf_matrix, label_names)
    except Exception as e:
        sys.stderr.write(f"Failed to compute confusion matrix: {e}")
