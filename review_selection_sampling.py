#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
import pprint
import shutil
import numpy as np
import pandas as pd
from sentiment_analysis_and_nlp.nlp_utils import detect_language
from useful_code_from_other_projects.utils import enable_max_pandas_display_size
from transformers import AutoTokenizer


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
                                            "game", "review_bombing_incident", "weighted_score", "developers",
                                            "publishers", "detailed_description", "title", "author_review_distribution",
                                            "user_score_distribution", "user_score", "critic_score", "num_user_ratings",
                                            "author_num_game_reviews", "author_average_score", "unhelpful_votes",
                                            "helpful_votes", "author_reviews_overall", "author_ratings_overall",
                                            "author_steam_level", "author_num_owned_games", "author_num_friends",
                                            "author_country_code", "author_last_online", "profile_created",
                                            "author_real_name", "username"],
                                   axis=1)

    # metacritic_rating = np.where(combined_df["rating"] < 3, "Nicht empfohlen", "Empfohlen")
    # combined_df.insert(2, "metacritic_rating", metacritic_rating)
    return combined_df


def create_combined_data_for_rb_incident(rb_incident_name: str, data_folder: pathlib.Path):
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
        filtered_metacritic_review_data = metacritic_reviews[(metacritic_reviews["rating"] < 3) & (
                metacritic_reviews["rating"] > 8)]
    elif use_very_positive_reviews:
        filtered_metacritic_review_data = metacritic_reviews[metacritic_reviews["rating"] > 8]
    else:
        filtered_metacritic_review_data = metacritic_reviews[metacritic_reviews["rating"] < 3]

    # combine all steam and metacritic data into one csv file with all the relevant information
    combined_review_df = combine_metacritic_steam_reviews(steam_reviews, filtered_metacritic_review_data, steam_info,
                                                          metacritic_info)
    combined_review_df.to_csv(Sub_Folder / f"combined_review_df_{rb_incident_name}.csv", index=False)


def remove_long_reviews(dataframe: pd.DataFrame) -> pd.DataFrame:
    # remove reviews with only one word/token (as they are most likely useless or at least too hard to interpret
    # correctly) as well as reviews with more than 512 tokens (as models such as BERT would have to truncate them
    # because of their character limit)
    # see https://stackoverflow.com/questions/72395380/how-to-drop-sentences-that-are-too-long-in-huggingface
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_review = tokenizer.tokenize(dataframe["review"])

    # TODO
    if (len(tokenized_review) > 512) | (len(tokenized_review) < 2):
        print(f"Review \"{tokenized_review}\" ({len(tokenized_review)}) was too long or short and therefore removed!")


def apply_stratified_sampling(review_data: pd.DataFrame, num_samples=10, random_seed=42):
    # see https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas/44115314
    target_sample_size = min(len(review_data.index), num_samples)
    randomly_sampled_reviews = review_data.sample(n=target_sample_size, random_state=random_seed)  # random sampling

    # TODO stratify per source platform, i.e. steam and metacritic
    # target_column = "source"

    # TODO stratify per game of this review bombing incident
    # target_column = "game_name_display"

    # TODO only works for steam, for Metacritic it would be "rating" (should probably add another column
    #  "rating_positive" (depending on "rating") as well for metacritic so both could be used the same here)
    # target_column = "steam_rating_positive"
    # stratified_sample = review_data.groupby(target_column, group_keys=False).apply(lambda x: x.sample(n=target_sample_size, random_state=random_seed))

    # random shuffle the stacked data, see https://stackoverflow.com/a/71948677
    randomly_sampled_reviews = randomly_sampled_reviews.sample(frac=1)
    return randomly_sampled_reviews


def select_reviews_for_label_studio(review_dataframe: pd.DataFrame):
    # remove reviews not written in english; unfortunately, this seems to remove some very short english reviews or
    # reviews with many links as well
    english_review_dataframe = review_dataframe[review_dataframe["review"].apply(lambda x: detect_language(x)).eq('en')]

    # only take reviews that contain at least 2-3 tokens and no more than 512 tokens (because of BERT token limit)
    # TODO
    # filtered_review_dataframe = remove_long_reviews(english_review_dataframe)

    # add a new column that contains the Metacritic score if it is a Metacritic review or the
    # "Recommended" / "Not recommended" label if it is a Steam review
    combined_rating = np.where(english_review_dataframe["source"] == "Steam",
                               np.where(english_review_dataframe["steam_rating_positive"], "Empfohlen", "Nicht empfohlen"),
                               english_review_dataframe["rating"].astype('Int64').astype("str") + " von 10")
    english_review_dataframe.insert(8, "combined_rating", combined_rating)

    # take subset with stratified sampling
    sampled_review_dataframe = apply_stratified_sampling(english_review_dataframe)
    return sampled_review_dataframe


if __name__ == "__main__":
    enable_max_pandas_display_size()
    if not OUTPUT_FOLDER.is_dir():
        OUTPUT_FOLDER.mkdir()

    review_bombing_incidents = {
        "Borderlands-Epic-Exclusivity": DATA_FOLDER / "Borderlands-Epic-Exclusivity",
        "Ukraine-Russia-Conflict": DATA_FOLDER / "Ukraine-Russia-Conflict",
        "Firewatch": DATA_FOLDER / "Firewatch",
    }

    ###### the name here needs to be updated for different review bombing incidents, see dictionary above ######
    review_bombing_name = "Borderlands-Epic-Exclusivity"
    # review_bombing_name = "Firewatch"
    data_path = review_bombing_incidents[review_bombing_name]
    label_studio_data_path = OUTPUT_FOLDER / f"{review_bombing_name}"

    combine_data_first = False
    if combine_data_first:
        create_combined_data_for_rb_incident(review_bombing_name, data_path)

    combined_review_dataframe = pd.read_csv(label_studio_data_path / f"combined_review_df_{review_bombing_name}.csv")
    sampled_review_df = select_reviews_for_label_studio(combined_review_dataframe)
    # save the final selected reviews in the output folder
    sampled_review_df.to_csv(label_studio_data_path / f"label_studio_df_{review_bombing_name}.csv", index=False)
