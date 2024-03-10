#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
import random
import numpy as np
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding, \
    create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from datasets import Dataset, load_dataset
import evaluate
import pandas as pd
import tensorflow as tf
import torch
import pprint
from classification.classifier_utils import set_random_seed
from sentiment_analysis_and_nlp.nlp_utils import detect_language
from useful_code_from_other_projects import utils
from useful_code_from_other_projects.utils import enable_max_pandas_display_size


INPUT_DATA = pathlib.Path(__file__).parent.parent / "annotated_data"


def preprocess_data(df: pd.DataFrame):
    pass


def classify_review_bombing(df: pd.DataFrame):
    pass


if __name__ == "__main__":
    enable_max_pandas_display_size()
    set_random_seed()

    # load relevant data
    combined_annotated_data = pd.read_csv(INPUT_DATA / "combined_final_annotation_all_projects.csv")
    pprint.pprint(combined_annotated_data.head())
