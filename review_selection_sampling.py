#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import pathlib
import pprint
import shutil
import numpy as np
import pandas as pd
from sentiment_analysis_and_nlp.nlp_utils import detect_language
from useful_code_from_other_projects.utils import enable_max_pandas_display_size, check_if_date_in_range
from transformers import AutoTokenizer
from datasets import Dataset


DATA_FOLDER = pathlib.Path(__file__).parent / "data_for_analysis_cleaned"
OUTPUT_FOLDER = pathlib.Path(__file__).parent / "data_for_labelstudio"


def load_data(path):
    concatenated_steam_info_df = pd.DataFrame()
    concatenated_steam_reviews_df = pd.DataFrame()
    concatenated_metacritic_info_df = pd.DataFrame()
    concatenated_metacritic_reviews_df = pd.DataFrame()

    for csv_file in path.glob("*.csv"):
        df = pd.read_csv(csv_file)  # use nrows=20 for faster testing
        if "info" in csv_file.name:
            if "steam" in csv_file.name:
                concatenated_steam_info_df = pd.concat([concatenated_steam_info_df, df], ignore_index=True,
                                                       verify_integrity=True)
            else:
                concatenated_metacritic_info_df = pd.concat([concatenated_metacritic_info_df, df], ignore_index=True,
                                                            verify_integrity=True)
        else:
            if "steam" in csv_file.name:
                concatenated_steam_reviews_df = pd.concat([concatenated_steam_reviews_df, df], ignore_index=True,
                                                          verify_integrity=True)
            else:
                concatenated_metacritic_reviews_df = pd.concat([concatenated_metacritic_reviews_df, df],
                                                               ignore_index=True, verify_integrity=True)

    return concatenated_steam_reviews_df, concatenated_steam_info_df, concatenated_metacritic_reviews_df, \
        concatenated_metacritic_info_df


def combine_metacritic_steam_reviews(reviews_steam: pd.DataFrame, reviews_metacritic: pd.DataFrame,
                                     game_info_steam: pd.DataFrame, game_info_metacritic: pd.DataFrame):
    """
    Combine metacritic and steam reviews into one unified dataframe. As steam reviews have far more features the
    Metacritic rows will contain a lot of empty values.
    """
    # pd.options.mode.chained_assignment = None  # disable some warnings

    # rename columns so the same content from both dataframes ends up in the same column
    reviews_steam = reviews_steam.rename(
        columns={"content": "review", "created_at_formatted": "review_date", "rating_positive": "steam_rating_positive",
                 "useful_score": "helpful_votes", "author_num_reviews": "author_num_game_reviews",
                 "author_username": "username"})
    game_info_steam = game_info_steam.rename(columns={"short_description": "game_description"})

    merged_reviews_df = pd.concat([reviews_steam, reviews_metacritic], axis=0, ignore_index=True)
    merged_game_info_df = pd.concat([game_info_steam, game_info_metacritic], axis=0, ignore_index=True)
    # merge the reviews and the general info for the game
    combined_df = merged_reviews_df.merge(merged_game_info_df, on=None, how='inner')

    # drop unnecessary columns
    combined_df = combined_df.drop(columns=["review_id", "created_at", "last_updated", "last_updated_formatted",
                                            "author_id", "comment_count", "platform", "profile_visibility",
                                            "profile_url", "game_id", "game_title", "price_euro",
                                            # the following are not useful for the annotation study in label studio:
                                            "review_bombing_incident", "weighted_score", "developers",
                                            "publishers", "detailed_description", "title", "author_review_distribution",
                                            "user_score_distribution", "user_score", "critic_score", "num_user_ratings",
                                            "author_num_game_reviews", "author_average_score", "unhelpful_votes",
                                            "helpful_votes", "author_reviews_overall", "author_ratings_overall",
                                            "author_steam_level", "author_num_owned_games", "author_num_friends",
                                            "author_country_code", "author_last_online", "profile_created",
                                            "author_real_name", "username"],
                                   axis=1)

    # add a new column for label studio that contains the Metacritic score if it is a Metacritic review or the
    # "Recommended" / "Not recommended" label if it is a Steam review
    combined_rating_display = np.where(combined_df["source"] == "Steam",
                                       np.where(combined_df["steam_rating_positive"], "Empfohlen", "Nicht empfohlen"),
                                       combined_df["rating"].astype('Int64').astype("str") + " von 10")
    combined_df.insert(8, "combined_rating_display", combined_rating_display)

    # also add a column that combines both the metacritic and the steam rating into one
    combined_rating = np.where(combined_df["source"] == "Steam", combined_df["combined_rating_display"],
                               np.where(combined_df["rating"] < 4, "Nicht empfohlen", "Empfohlen"))
    combined_df.insert(8, "combined_rating", combined_rating)

    # remove completely empty rows
    combined_df_new = combined_df.dropna(how="all")
    return combined_df_new


def create_combined_data_for_rb_incident(rb_incident_name: str, data_folder: pathlib.Path, metacritic_rb_start: str,
                                         metacritic_rb_end: str):
    print(f"Selecting reviews for \"{rb_incident_name}\" from path {data_folder} ...\n")
    # create a subfolder for this review bombing incident if it doesn't exist yet
    Sub_Folder = OUTPUT_FOLDER / rb_incident_name
    if not Sub_Folder.is_dir():
        Sub_Folder.mkdir()
    else:
        print("WARNING: Subfolder already exists!")
        answer = input(f"Do you want to overwrite the existing folder for \"{rb_incident_name}\"? [y/n]\n")
        if str.lower(answer) == "y" or str.lower(answer) == "yes":
            shutil.rmtree(Sub_Folder)
            Sub_Folder.mkdir()
        else:
            return

    # load all relevant files as pandas dataframes and then select reviews for label studio by sampling
    steam_reviews, steam_info, metacritic_reviews, metacritic_info = load_data(path=data_folder)

    # steam_reviews = remove_linebreaks_from_pd_cells(steam_reviews, column_name="content")

    # take only very negative or very positive metacritic reviews (or both) depending on the type of review bombing
    # (i.e. for the ukraine-russia-review bombing we take both very negative and very positive reviews)
    rb_type = metacritic_reviews.at[0, "review_bomb_type"]
    if rb_type == "negativ":
        use_very_negative_reviews = True
        use_very_positive_reviews = False
    elif rb_type == "positiv":
        use_very_negative_reviews = False
        use_very_positive_reviews = True
    else:
        use_very_negative_reviews = True
        use_very_positive_reviews = True

    if use_very_negative_reviews and use_very_positive_reviews:
        # TODO take < 4 instead of < 3 to get more reviews ?
        filtered_metacritic_review_data = metacritic_reviews[(metacritic_reviews["rating"] < 4) | (
                metacritic_reviews["rating"] > 8)]
    elif use_very_positive_reviews:
        filtered_metacritic_review_data = metacritic_reviews[metacritic_reviews["rating"] > 8]
    else:
        filtered_metacritic_review_data = metacritic_reviews[metacritic_reviews["rating"] < 4]

    # make sure the metacritic reviews are all in the time range of the corresponding review bombing incident (steam
    # reviews are already retrieved correctly from Steam)
    filtered_metacritic_review_data = filtered_metacritic_review_data[
        filtered_metacritic_review_data["review_date"].apply(
            lambda review_date: check_if_date_in_range(review_date, metacritic_rb_start, metacritic_rb_end)).eq(True)]

    # combine all steam and metacritic data into one csv file with all the relevant information
    combined_review_df = combine_metacritic_steam_reviews(steam_reviews, filtered_metacritic_review_data, steam_info,
                                                          metacritic_info)
    combined_review_df.to_csv(Sub_Folder / f"combined_review_df_{rb_incident_name}.csv", index=False)


def remove_short_long_reviews(dataframe: pd.DataFrame) -> pd.DataFrame:
    # remove reviews with less than 3 tokens (as they are most likely useless or at least too hard to interpret
    # correctly) as well as reviews with more than 512 tokens (as models such as BERT would have to truncate them
    # because of their character limit)
    # see https://stackoverflow.com/questions/72395380/how-to-drop-sentences-that-are-too-long-in-huggingface
    dataset = Dataset.from_pandas(dataframe)
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # TODO use spacy instead to check for at least 2 words instead of tokens ? and remove punctuation first
    # minimum is 5 instead of 3 because the tokenizer adds two additional tokens (a start and end token) automatically
    min_tokens = 5
    # max input size for "bert-base-uncased" is 512:
    max_tokens = tokenizer.max_model_input_sizes[checkpoint]

    def tokenize_function(df):
        return tokenizer(df["review"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(f"Num rows before: {len(tokenized_datasets)}")
    tokenized_datasets = tokenized_datasets.filter(lambda df: min_tokens <= len(df['input_ids']) <= max_tokens)
    print(f"Num rows after removing too long or too short reviews: {len(tokenized_datasets)}")

    filtered_df = tokenized_datasets.to_pandas()
    filtered_df = filtered_df.drop(columns=["token_type_ids", "attention_mask"], axis=1)  # "input_ids",
    return filtered_df


def apply_random_sampling(review_data: pd.DataFrame, num_samples=25, random_seed=42):
    target_sample_size = min(len(review_data.index), num_samples)
    randomly_sampled_reviews = review_data.sample(n=target_sample_size, random_state=random_seed)  # random sampling
    return randomly_sampled_reviews


def apply_stratified_sampling(review_data: pd.DataFrame, num_samples=250, random_seed=42):
    # see https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas/44115314
    #################################
    print("\n#################################\nReview Dataframe - Overview:")
    for column_name in ["source", "game_name_display", "combined_rating"]:
        print(f"\nColumn: {review_data[column_name].value_counts()}")
        ratio = review_data[column_name].value_counts(normalize=True)
        print(f"ratio_percentage: {ratio.round(4) * 100}")

    grouped_ratio = review_data.value_counts(["game_name_display", "source", "combined_rating"])
    print(f"\nGrouped Ratio: {grouped_ratio}")
    print("\n#################################\n")
    #################################

    # TODO would this way of sampling be better overall for larger num_samples ? Test and compare!
    # This would be correctly stratified, but since there are a lot fewer Metacritic reviews than Steam reviews (and
    # often one game has far more reviews than another), we need to make sure the classes are somewhat balanced
    """
    stratified_sample = review_data.groupby(["game_name_display", "source", "combined_rating"], group_keys=False) \
        .apply(lambda x: x.sample(frac=min(num_samples / len(review_data), 1), random_state=random_seed))
    """

    """
    # use stratified sampling with target sample size: https://stackoverflow.com/a/54722093
    # not working correctly:
    stratified_sample = review_data.groupby(["game_name_display", "source", "combined_rating"], group_keys=False).apply(
        lambda x: x.sample(frac=int(np.rint(num_samples * len(x) / len(review_data))), random_state=42))
    """

    def sample_func(x):
        N = min(len(x), sample_size_per_group)
        return x.sample(n=N, random_state=random_seed)

    # divide by the group size and round up to get approximately the wanted num_samples stratified by grouping
    sample_size_per_group = math.ceil(num_samples / grouped_ratio.size)  # ceil to make sure we have at least 1
    stratified_sample = review_data.groupby(["game_name_display", "source", "combined_rating"],
                                            group_keys=False).apply(lambda x: sample_func(x))

    # TODO also groupby and stratify by "review_date" to make sure that the selected reviews are somewhat evenly
    #  distributed across the entire review bombing timespan instead of only the first 2 or 3 days ?

    if "__index_level_0__" in stratified_sample:
        stratified_sample = stratified_sample.drop(columns=["__index_level_0__"], axis=1)

    stratified_grouped_ratio = stratified_sample.value_counts(["game_name_display", "source", "combined_rating"])
    print(f"\nGrouped Ratio after sampling: {stratified_grouped_ratio}")

    # random shuffle the stacked data at the end, see https://stackoverflow.com/a/71948677
    randomly_sampled_reviews = stratified_sample.sample(frac=1).reset_index(drop=True)
    return randomly_sampled_reviews


def select_reviews_for_label_studio(review_dataframe: pd.DataFrame):
    # remove reviews not written in english; unfortunately, this seems to remove some very short english reviews or
    # reviews with many links as well
    english_review_dataframe = review_dataframe[review_dataframe["review"].apply(lambda x: detect_language(x)).eq('en')]
    print(f"Removed {len(review_dataframe) - len(english_review_dataframe)} non-english reviews")

    # remove all rows with duplicate reviews to make sure all reviews annotated in label studio will be different
    no_duplicates_review_dataframe = english_review_dataframe.drop_duplicates(subset=["review"])
    print(f"Removed {len(english_review_dataframe) - len(no_duplicates_review_dataframe)} rows with duplicate reviews")

    # only take reviews that contain at least 2 tokens and no more than 512 tokens (because of BERT token limit)
    filtered_review_dataframe = remove_short_long_reviews(no_duplicates_review_dataframe)

    # take subset with stratified sampling
    sampled_review_dataframe = apply_stratified_sampling(filtered_review_dataframe)
    return sampled_review_dataframe


if __name__ == "__main__":
    enable_max_pandas_display_size()
    if not OUTPUT_FOLDER.is_dir():
        OUTPUT_FOLDER.mkdir()

    review_bombing_incidents = {
        "Assassins-Creed-Unity": DATA_FOLDER / "Assassins-Creed-Unity",
        # "Bethesda-Creation-Club": DATA_FOLDER / "Bethesda-Creation-Club",
        "Borderlands-Epic-Exclusivity": DATA_FOLDER / "Borderlands-Epic-Exclusivity",
        # "Crusader-Kings-II-Deus-Vult": DATA_FOLDER / "Crusader-Kings-II-Deus-Vult",
        "Firewatch": DATA_FOLDER / "Firewatch",
        "GrandTheftAutoV-OpenIV": DATA_FOLDER / "GrandTheftAutoV-OpenIV",
        "Metro-Epic-Exclusivity": DATA_FOLDER / "Metro-Epic-Exclusivity",
        "Mortal-Kombat-11": DATA_FOLDER / "Mortal-Kombat-11",
        # "Overwatch-2": DATA_FOLDER / "Overwatch-2",
        # "Skyrim-Paid-Mods": DATA_FOLDER / "Skyrim-Paid-Mods",
        # "Superhot-VR": DATA_FOLDER / "Superhot-VR",
        "The-Long-Dark-GeForce-Now": DATA_FOLDER / "The-Long-Dark-GeForce-Now",
        "TotalWar-Rome-II": DATA_FOLDER / "TotalWar-Rome-II",
        "Ukraine-Russia-Conflict": DATA_FOLDER / "Ukraine-Russia-Conflict",
    }

    ###### the start and end date of the review bombing must be updated for every review bombing incident! ######
    # see the dictionary at the top of the "map_cleanup_downloaded_data.py" file
    metacritic_start_date = "24.02.2022"
    metacritic_end_date = "16.04.2022"

    ###### the name here needs to be updated for different review bombing incidents, see dictionary above ######
    review_bombing_name = "Ukraine-Russia-Conflict"
    data_path = review_bombing_incidents[review_bombing_name]
    label_studio_data_path = OUTPUT_FOLDER / f"{review_bombing_name}"

    combine_data_first = False
    if combine_data_first:
        create_combined_data_for_rb_incident(review_bombing_name, data_path, metacritic_start_date, metacritic_end_date)

    combined_review_dataframe = pd.read_csv(label_studio_data_path / f"combined_review_df_{review_bombing_name}.csv")
    sampled_review_df = select_reviews_for_label_studio(combined_review_dataframe)
    sampled_review_df.to_csv(label_studio_data_path / f"label_studio_df_{review_bombing_name}.csv", index=False)
