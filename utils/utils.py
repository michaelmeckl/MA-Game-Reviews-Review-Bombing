"""
Code below used / needed at different locations in the codebase.
"""

import json
import numpy as np
from dateutil import parser
import pandas as pd
from smart_open import open


def check_if_date_in_range(date: str, date_start: str, date_end: str):
    """
    Checks if a given date is in the time span between the provided start and end date (both inclusive).
    """
    parsed_date = parser.parse(date, dayfirst=True)
    parsed_start_date = parser.parse(date_start, dayfirst=True)
    parsed_end_date = parser.parse(date_end, dayfirst=True)
    is_in_range = True if (parsed_start_date <= parsed_date) & (parsed_date <= parsed_end_date) else False
    return is_in_range


def convert_to_int(x):
    """
    Useful for parsing Metacritic, as bigger numbers are abbreviated like 1.5k instead of 1500 for example.
    This method converts it correctly.
    """
    if type(x) == int:
        return x
    if 'k' in x:
        if len(x) > 1:
            converted = float(x.replace('k', '')) * 1000
            return int(converted)
        return 1000
    else:
        return int(x)


def write_json_to_file(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def read_json_from_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
        return data


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


def remove_whitespaces_from_cells(dataframe: pd.DataFrame):
    # remove all trailing and leading whitespaces
    dataframe.columns = dataframe.columns.str.strip()
    obj_cols = dataframe.select_dtypes(['object']).columns
    dataframe[obj_cols] = dataframe[obj_cols].apply(lambda x: x.str.strip())
    return dataframe


def enable_max_pandas_display_size():
    pd.options.display.width = 0


def remove_duplicates(df: pd.DataFrame, column: str = None) -> pd.DataFrame:
    without_duplicates = df.drop_duplicates() if column is None else df.drop_duplicates(subset=[column])
    # duplicate_entries = df[df.duplicated(subset=[column])]
    print(f"Removed {len(df) - len(without_duplicates)} duplicate entries")
    return without_duplicates


def combine_identical_dataframes(df_list: list[pd.DataFrame], drop_duplicates=True, drop_column="id", sort_df=True,
                                 sort_column=None):
    """
    Combines a list of dataframes with the same columns. Optionally drops duplicate entries and sorts the
    resulting dataframe.
    """
    combined_dataframe = pd.concat(df_list, ignore_index=True)
    if drop_duplicates:
        combined_dataframe = remove_duplicates(df=combined_dataframe, column=drop_column)  # .reset_index(drop=True)
    if sort_df and sort_column is not None:
        combined_dataframe = combined_dataframe.sort_values(by=sort_column, ascending=False)
    return combined_dataframe


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


def concat_generators(*iterables):
    # Concatenates an arbitrary number of generators (note: this returns the output for each generator in order!).
    # See https://stackoverflow.com/a/47592164
    for iterable in iterables:
        yield from iterable


def cosine_similarity(u, v):
    # Calculates the cosine similarity between two vectors
    # Taken from https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
