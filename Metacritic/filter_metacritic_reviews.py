#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pathlib
import pandas as pd
from sentiment_analysis_and_nlp.language_detection import detect_language
from utils import utils


def search_keywords(text: str, keywords: list[str]):
    # TODO basic text normalization like stopword removal / stemming / etc. is missing here !
    text = text.lower()
    if any(word.lower() in text for word in keywords):
        return True
    else:
        return False


def filter_extracted_reviews(df: pd.DataFrame, search_for_keywords=True):
    # only keep rows where the rating is below 3 as these are most likely to be a Review-Bombing Review
    highly_negative_reviews = df[df["rating"] < 3]
    print(f"Ratings: {highly_negative_reviews['rating'].value_counts()}")

    # ! important as otherwise we would only get the last sentence for some cells
    highly_negative_reviews = utils.remove_linebreaks_from_pd_cells(highly_negative_reviews, column_name="review")

    # remove text in languages other than english or german with the langdetect module
    language_filtered_df = highly_negative_reviews[
        highly_negative_reviews["review"].apply(lambda x: detect_language(x)).eq('en', 'de')]

    if search_for_keywords:
        # search for specific keywords in the review text and keep only reviews that match
        # TODO which keywords appear often in "Review - Bombing - Reviews" ??
        # probably game-specific, e.g. for cyberpunk: "lie", "ukraine", "russia", "politic", "labor of love", "award"
        keyword_list = ["lie", "ukraine", "russia", "politic", "labor of love", "award"]
        keyword_filtered_df = language_filtered_df[
            language_filtered_df["review"].apply(lambda review: search_keywords(review, keyword_list)).eq(True)]
        return keyword_filtered_df
    else:
        return language_filtered_df


# don't remove duplicates as they might be useful in detecting fake reviews / review bombing ?
if __name__ == "__main__":
    REVIEW_DATA = pathlib.Path(__file__).parent / "metacritic_data"

    for game_folder in os.listdir(REVIEW_DATA):
        game_folder_path = REVIEW_DATA / game_folder
        for filename in os.listdir(game_folder_path):
            if filename.startswith("user_reviews"):
                print(f"Processing file \"{filename}\" ...")
                review_df = pd.read_csv(game_folder_path / filename)

                filtered_df = filter_extracted_reviews(review_df, search_for_keywords=False)
                print(f"\nNum rows before filtering: {len(review_df)}\nNum rows after filtering: {len(filtered_df)}")
                filtered_df.to_csv(game_folder_path / f"filtered_metacritic_{filename}", index=False)
