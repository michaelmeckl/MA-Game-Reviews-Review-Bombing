#!/usr/bin/python3
# -*- coding:utf-8 -*-

import pathlib
from enum import Enum

RANDOM_SEED = 42

INPUT_DATA_FOLDER = pathlib.Path(__file__).parent.parent
MODEL_FOLDER = pathlib.Path(__file__).parent / "trained_models"

# the colum names that contain the annotation results
annotation_questions = ["is-review-bombing", "is-rating-game-related", "criticism-praise-game-content",
                        "criticism-praise-developer-publisher", "criticism-praise-ideological",
                        "criticism-praise-political"]


class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
