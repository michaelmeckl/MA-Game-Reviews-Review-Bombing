#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import pprint
from datetime import datetime
import pathlib
import numpy as np
import pandas as pd
from useful_code_from_other_projects.utils import enable_max_pandas_display_size

DATA_FOLDER = pathlib.Path(__file__).parent


def add_combined_rating_column(input_df: pd.DataFrame):
    # Add a new column that contains the Metacritic score if it is a Metacritic review or the
    # "Recommended" / "Not recommended" label if it is a Steam review

    # ! since all steam reviews are currently "Not recommended" we skip the check for the column value
    new_rating = np.where(input_df["source"] == "Steam", "Nicht empfohlen",
                          input_df["rating"].astype('Int64').astype("str") + " von 10")
    # new_rating = new_rating.astype('str', copy=False)
    input_df.insert(2, "combined_rating", new_rating)


if __name__ == "__main__":
    enable_max_pandas_display_size()

    filename = "combined_steam_metacritic_df_Borderlands_edited"
    input_data = pd.read_csv(DATA_FOLDER / f"{filename}.csv")
    add_combined_rating_column(input_data)
    input_data.to_csv(DATA_FOLDER / f"labelstudio_{filename}.csv", index=False)
