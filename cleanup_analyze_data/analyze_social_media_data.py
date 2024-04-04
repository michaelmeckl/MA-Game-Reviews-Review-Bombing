#!/usr/bin/python
# -*- coding: utf-8 -*-

import pprint
import pathlib
import numpy as np
import pandas as pd
from utils.utils import enable_max_pandas_display_size


def analyze_twitter_data(twitter_data_df: pd.DataFrame):
    pass


if __name__ == "__main__":
    enable_max_pandas_display_size()

    # load data
    DATA_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis"
    # steam_review_data = pd.read_csv(DATA_FOLDER / "steam" / "steam_user_reviews_cyberpunk_2077.csv")

    # TODO analyze temporal and other anomalies in the data (especially Twitter & Reddit)
    #  -> also aggregate useful information such as the number tweets in a certain time period or the average sentiment
