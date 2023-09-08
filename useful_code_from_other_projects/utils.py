"""
This folder serves as a repository for all kinds of potentially useful code from previous projects.

Code below used / needed at different locations in the codebase.
"""
import shutil
import os
import glob
import json
import pathlib
from collections import defaultdict
from datetime import datetime
from dateutil import parser
import time
from typing import Any
import pandas as pd
from smart_open import open


def count_invocation(f):
    """ Decorator to log how often a function is called. """
    n = 0

    def decorated(*args, **kwargs):
        nonlocal n
        if n == 0:
            print("The function", f.__name__, " has been called once at ", time.asctime())
        else:
            print("The function", f.__name__, " has been called ", n + 1, " times at ", time.asctime())
        n += 1
        return f(*args, **kwargs)

    return decorated


# Usage of defaultdict example (check for how many strings in dict a condition applies)
def defaultdict_example():
    dictionary = {
        "a": ["testM", "Mtest", "_test"],
        "b": ["test1", "test2", "mtest"]
    }

    result_dict = defaultdict(int)
    for key, value in dictionary.items():
        for item in value:
            if item.lower().startswith("m"):
                result_dict[key] += 1


# Example: find all files with a specific file ending in a directory and replace file ending
def convert_to_png():
    OUTPUT_FOLDER = pathlib.Path(__file__).parent / "new_images"
    FILE_ENDING = ".webp"

    # create new subfolder for the new images
    if not OUTPUT_FOLDER.is_dir():
        OUTPUT_FOLDER.mkdir()
    else:
        print("WARNING: Output folder already exists!")
        # remove old folder if there is already one
        shutil.rmtree(OUTPUT_FOLDER)
        OUTPUT_FOLDER.mkdir()

    counter = 0
    # Find all .webp images in the current directory
    for original_image_path in glob.glob(f"./*{FILE_ENDING}"):
        counter += 1
        # alternatively splitext('path') could have been used
        filename = (os.path.split(original_image_path)[-1])
        filename_without_type = filename.split(".")[0]
        # convert to .png
        output_path = f"{OUTPUT_FOLDER}/{filename_without_type}.png"
        # save at new output path
        # ...
    print(f"Found and converted {counter} .webp images in this directory.")


# Date parsing example
def parse_date(original_date):
    """
    Figures out the format of a given date and converts it into the specified form.
    see https://stackoverflow.com/questions/2265357/parse-date-string-and-change-format
    """
    original_date = parser.parse(original_date)
    new_date = original_date.strftime('%d.%m.%Y %H:%M:%S')
    return new_date


def write_json_to_file(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def read_json_from_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
        return data


def log_data_as_csv(data: dict[Any, Any]):
    """
    Save the given data as a csv file in a folder called "data".
    """
    save_location = pathlib.Path(__file__).parent / "data"

    # `**data` unpacks the given dictionary as key-value pairs
    df = pd.DataFrame({'date': datetime.now(), **data}, index=[0])
    df.to_csv(save_location, sep=";", index=False)


def remove_linebreaks_from_pd_cells(dataframe: pd.DataFrame, column_name=None):
    """
    Removes linebreaks in the cell contents of a pandas dataframe (as they could mess up a csv file for example).
    This method does not work in-place at the moment.
    """
    replacement_regex = r'\r+|\n+|\t+'

    if column_name is not None:
        dataframe.loc[:, column_name] = dataframe[column_name].str.replace(replacement_regex, ' ', regex=True)
        return dataframe
    else:
        return dataframe.replace(replacement_regex, ' ', regex=True)


def enable_max_pandas_display_size():
    pd.options.display.width = 0


def remove_duplicates(df: pd.DataFrame, column: str = None) -> pd.DataFrame:
    without_duplicates = df.drop_duplicates() if column is None else df.drop_duplicates(subset=[column])
    # duplicate_entries = df[df.duplicated(subset=[column])]
    print(f"Removed {len(df) - len(without_duplicates)} duplicate entries")
    return without_duplicates


def compare_pandas_dataframes(df_1: pd.DataFrame, df_2: pd.DataFrame, merge_column: str, df_1_name="df_1",
                              df_2_name="df_2", write_to_csv=True):
    """
    The function expects both dataframes to have the same columns.
    Returns the overlapping entries in both dataframes as a new dataframe.
    """
    df_1_2_overlap = df_1.merge(df_2[[merge_column]], how="inner", on=merge_column)
    df_1_unique = df_1[~df_1[merge_column].isin(df_2[merge_column])]
    df_2_unique = df_2[~df_2[merge_column].isin(df_1[merge_column])]

    """
    merged_df = df_1.merge(df_2, on=[merge_column], how='outer', indicator=True)
    df_1_2_overlap = merged_df[merged_df['_merge'] == 'both']
    df_1_unique = merged_df[merged_df['_merge'] == 'left_only']
    df_2_unique = merged_df[merged_df['_merge'] == 'right_only']
    """

    print(f"\nunique: {len(df_1_unique)}, overlap: {len(df_1_2_overlap)}, overall: {len(df_1)}")
    print(f"unique: {len(df_2_unique)}, overlap: {len(df_1_2_overlap)}, overall: {len(df_2)}")
    print(f"Overlapping rows between the dataframes: {len(df_1_2_overlap)}")
    print(df_1_2_overlap.head())

    if write_to_csv:
        df_1_unique.to_csv(f"{df_1_name}_unique.csv", index=False)
        df_2_unique.to_csv(f"{df_2_name}_unique.csv", index=False)
        df_1_2_overlap.to_csv(f"{df_1_name}-{df_2_name}-overlap.csv", index=False)

    return df_1_2_overlap
