#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
import pprint
import re
import shutil
import pandas as pd
from cleanup_analyze_data.review_bombing_incidents_info import review_bombing_incidents
from sentiment_analysis_and_nlp.language_detection import detect_language
from utils.utils import enable_max_pandas_display_size, concat_generators, check_if_date_in_range

DATA_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis"
STEAM_DATA_FOLDER = DATA_FOLDER / "steam"
METACRITIC_DATA_FOLDER = DATA_FOLDER / "metacritic"
TWITTER_DATA_FOLDER = DATA_FOLDER / "tweets"
REDDIT_DATA_FOLDER = DATA_FOLDER / "reddit"

OUTPUT_FOLDER_REVIEWS = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "reviews"
OUTPUT_FOLDER_POSTS = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "posts"


###############################################################################
# the keys in this dictionary must map to the file names in the Metacritic and Steam folders in
# 'data_for_analysis' without the underscores and hyphens (case doesn't matter)
game_names_mapping = {
    "assassins creed unity.csv": {
        "game": "Assassins Creed Unity",
        "game_name_display": "Assassin's Creed Unity",
    },
    "assassin s creed unity.csv": {
        "game": "Assassins Creed Unity",
        "game_name_display": "Assassin's Creed Unity",
    },
    "borderlands.csv": {
        "game": "Borderlands",
        "game_name_display": "Borderlands",
    },
    "borderlands GOTY.csv": {
        "game": "Borderlands",
        "game_name_display": "Borderlands GOTY",
    },
    "borderlands GOTY enhanced.csv": {
        "game": "Borderlands",
        "game_name_display": "Borderlands GOTY Enhanced",
    },
    "borderlands 2.csv": {
        "game": "Borderlands 2",
        "game_name_display": "Borderlands 2",
    },
    "borderlands the pre sequel.csv": {
        "game": "Borderlands: The Pre-Sequel",
        "game_name_display": "Borderlands: The Pre-Sequel",
    },
    "crusader kings ii.csv": {
        "game": "Crusader Kings II",
        "game_name_display": "Crusader Kings II",
    },
    # ! important to not use ".csv" here to find all the Cyberpunk 2077 files from different incidents
    "cyberpunk 2077": {
        "game": "Cyberpunk 2077",
        "game_name_display": "Cyberpunk 2077",
    },
    "fallout 4.csv": {
        "game": "Fallout 4",
        "game_name_display": "Fallout 4",
    },
    "firewatch.csv": {
        "game": "Firewatch",
        "game_name_display": "Firewatch",
    },
    "frostpunk.csv": {
        "game": "Frostpunk",
        "game_name_display": "Frostpunk",
    },
    "grand theft auto v.csv": {
        "game": "Grand Theft Auto V",
        "game_name_display": "Grand Theft Auto V",
    },
    "gwent the witcher card game.csv": {
        "game": "GWENT: The Witcher Card Game",
        "game_name_display": "GWENT: The Witcher Card Game",
    },
    "hogwarts legacy": {
        "game": "Hogwarts Legacy",
        "game_name_display": "Hogwarts Legacy",
    },
    "metro 2033.csv": {
        "game": "Metro 2033",
        "game_name_display": "Metro 2033",
    },
    "metro 2033 redux.csv": {
        "game": "Metro 2033 Redux",
        "game_name_display": "Metro 2033 Redux",
    },
    "metro last light.csv": {
        "game": "Metro: Last Light",
        "game_name_display": "Metro: Last Light",
    },
    "metro last light redux.csv": {
        "game": "Metro: Last Light Redux",
        "game_name_display": "Metro: Last Light Redux",
    },
    "mortal kombat 11.csv": {
        "game": "Mortal Kombat 11",
        "game_name_display": "Mortal Kombat 11",
    },
    "no mans sky.csv": {
        "game": "No Mans Sky",
        "game_name_display": "No Man's Sky",
    },
    "no man s sky.csv": {
        "game": "No Mans Sky",
        "game_name_display": "No Man's Sky",
    },
    "overwatch 2": {
        "game": "Overwatch 2",
        "game_name_display": "Overwatch 2",
    },
    "s t a l k e r shadow of chernobyl.csv": {
        "game": "STALKER: Shadow of Chernobyl",
        "game_name_display": "S.T.A.L.K.E.R.: Shadow of Chernobyl",
    },
    "s t a l k e r clear sky.csv": {
        "game": "STALKER: Clear Sky",
        "game_name_display": "S.T.A.L.K.E.R.: Clear Sky",
    },
    "s t a l k e r call of pripyat.csv": {
        "game": "STALKER: Call of Pripyat",
        "game_name_display": "S.T.A.L.K.E.R.: Call of Pripyat",
    },
    "superhot vr.csv": {
        "game": "Superhot VR",
        "game_name_display": "Superhot VR",
    },
    "the elder scrolls v skyrim.csv": {
        "game": "The Elder Scrolls V: Skyrim",
        "game_name_display": "The Elder Scrolls V: Skyrim",
    },
    "the elder scrolls v skyrim special edition.csv": {
        "game": "The Elder Scrolls V: Skyrim",
        "game_name_display": "The Elder Scrolls V: Skyrim Special Edition",
    },
    "the long dark.csv": {
        "game": "The Long Dark",
        "game_name_display": "The Long Dark",
    },
    "the witcher 2 assassins of kings.csv": {
        "game": "The Witcher 2: Assassins of Kings",
        "game_name_display": "The Witcher 2: Assassins of Kings",
    },
    "the witcher 2 assassins of kings enhanced edition.csv": {
        "game": "The Witcher 2: Assassins of Kings",
        "game_name_display": "The Witcher 2: Assassins of Kings Enhanced Edition",
    },
    "the witcher 3 wild hunt.csv": {
        "game": "The Witcher 3: Wild Hunt",
        "game_name_display": "The Witcher 3: Wild Hunt",
    },
    "the witcher 3 wild hunt complete edition.csv": {
        "game": "The Witcher 3: Wild Hunt",
        "game_name_display": "The Witcher 3: Wild Hunt - Complete Edition",
    },
    "the witcher.csv": {
        "game": "The Witcher",
        "game_name_display": "The Witcher",
    },
    "the witcher enhanced edition.csv": {
        "game": "The Witcher",
        "game_name_display": "The Witcher Enhanced Edition",
    },
    "thronebreaker the witcher tales.csv": {
        "game": "Thronebreaker: The Witcher Tales",
        "game_name_display": "Thronebreaker: The Witcher Tales",
    },
    "total war rome ii.csv": {
        "game": "Total War: ROME II",
        "game_name_display": "Total War: ROME II",
    },
    "total war rome ii emperor edition.csv": {
        "game": "Total War: ROME II",
        "game_name_display": "Total War: ROME II - Emperor Edition",
    },
}

###############################################################################


"""
def select_relevant_data_files(games_terms, select_only_reviews=False):
    # select all the relevant csv files for the given games with glob
    relevant_steam_files = []
    relevant_metacritic_files = []
    for game in games_terms:
        if select_only_reviews:
            pattern = f"*user_reviews*{game}*.csv"
        else:
            pattern = f"*{game}*.csv"
        steam_files = [f for f in STEAM_DATA_FOLDER.glob(pattern)]
        metacritic_files = [f for f in METACRITIC_DATA_FOLDER.glob(pattern)]
        relevant_steam_files.extend(steam_files)
        relevant_metacritic_files.extend(metacritic_files)

    print(f"{len(relevant_steam_files)} Steam files found:")
    pprint.pprint(relevant_steam_files)
    print(f"{len(relevant_metacritic_files)} Metacritic files found:")
    pprint.pprint(relevant_metacritic_files)
    return relevant_steam_files, relevant_metacritic_files
"""


def add_game_name_to_game_files():
    """
    Add two additional columns for the game name: "game_name_display" for the actual name that is shown in Label
    Studio and "game" for easier mapping (e.g. to map "Borderlands" and "Borderlands GOTY" to the same game).
    This way the game_info and user_reviews files in metacritic and steam can be matched and merged accordingly.
    Also add another column "source" to differentiate between the review platforms.
    """
    for file in concat_generators(METACRITIC_DATA_FOLDER.glob("*.csv"), STEAM_DATA_FOLDER.glob("*.csv")):
        # find the correct dict entry by checking if the filename contains the key string
        for key in game_names_mapping.keys():
            # unify file names by removing _, - or whitespace
            file_name_cleaned = re.sub(r'[_-]+', ' ', file.name).lower()
            if key.lower() in file_name_cleaned:
                df = pd.read_csv(file)
                if "source" not in df:
                    source_value = "Steam" if file.parent.name == "steam" else "Metacritic"
                    df.insert(0, "source", [source_value] * len(df))
                if "game_name_display" not in df:
                    game_name_display = game_names_mapping[key]["game_name_display"]
                    df.insert(0, "game_name_display", [game_name_display] * len(df))
                if "game" not in df:
                    game_name = game_names_mapping[key]["game"]
                    df.insert(0, "game", [game_name] * len(df))

                # overwrite the old file
                df.to_csv(file, index=False)


def add_rb_information_to_game_files(rb_incident_name):
    """
    Map each file in the steam and metacritic folder to the corresponding review bombing incident by adding
    additional columns for the rb incident (must be the same name as in the twitter and reddit files for mapping!).
    """
    print(f"\n##########################\nUpdating files for \"{rb_incident_name}\" ...\n")

    # create a subfolder for this review bombing incident if it doesn't exist yet
    Sub_Folder = OUTPUT_FOLDER_REVIEWS / rb_incident_name
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

    rb_information = review_bombing_incidents[rb_incident_name]
    games_title_terms = rb_information["games_title_terms"]
    affected_games = rb_information["affected_games"]
    review_bomb_type = rb_information["review_bomb_type"]
    review_bomb_reason = rb_information["review_bomb_reason"]
    review_bomb_time = rb_information["review_bomb_time"]

    relevant_steam_files = []
    relevant_metacritic_files = []
    for term in games_title_terms:
        pattern = f"{term}.csv"
        steam_files = [f for f in STEAM_DATA_FOLDER.glob(pattern)]
        metacritic_files = [f for f in METACRITIC_DATA_FOLDER.glob(pattern)]
        relevant_steam_files.extend(steam_files)
        relevant_metacritic_files.extend(metacritic_files)

    print(f"{len(relevant_steam_files)} Steam files found:")
    pprint.pprint(relevant_steam_files)
    print(f"{len(relevant_metacritic_files)} Metacritic files found:")
    pprint.pprint(relevant_metacritic_files)

    # add the new columns and save updated data to new folder
    for file in relevant_steam_files + relevant_metacritic_files:
        df = pd.read_csv(file)
        df.insert(3, "review_bomb_reason", [review_bomb_reason] * len(df))
        df.insert(3, "review_bomb_time", [review_bomb_time] * len(df))
        df.insert(3, "review_bomb_type", [review_bomb_type] * len(df))
        df.insert(3, "affected_games", [affected_games] * len(df))
        df.insert(3, "review_bombing_incident", [rb_incident_name] * len(df))
        df.to_csv(Sub_Folder / f"{file.stem}_updated.csv", index=False)


def add_rb_information_to_social_media_files(rb_incident_name):
    """
    Map each file in the twitter and reddit folder to the corresponding review bombing incident.
    """
    print(f"\n##########################\nUpdating files for \"{rb_incident_name}\" ...\n")
    rb_information = review_bombing_incidents[rb_incident_name]
    social_media_title_terms = rb_information["social_media_title_terms"]
    affected_games = rb_information["affected_games"]

    relevant_twitter_files = []
    relevant_reddit_files = []
    for term in social_media_title_terms:
        pattern = f"{term}.csv"
        twitter_files = [f for f in TWITTER_DATA_FOLDER.glob(pattern)]
        reddit_files = [f for f in REDDIT_DATA_FOLDER.glob(pattern)]
        relevant_twitter_files.extend(twitter_files)
        relevant_reddit_files.extend(reddit_files)

    print(f"{len(relevant_twitter_files)} Twitter files found:")
    pprint.pprint(relevant_twitter_files)
    print(f"{len(relevant_reddit_files)} Reddit files found:")
    pprint.pprint(relevant_reddit_files)

    twitter_dataframes = []
    reddit_comments_dataframes = []
    reddit_submissions_dataframes = []
    # add the new columns and save updated dataframe to the correct list
    for file in relevant_twitter_files + relevant_reddit_files:
        df = pd.read_csv(file)
        df.insert(3, "affected_games", [affected_games] * len(df))
        df.insert(3, "review_bombing_incident", [rb_incident_name] * len(df))
        # add the social media data source (Twitter or Reddit) as an additional column
        if file.parent.name == "reddit":
            df.insert(3, "source", ["Reddit"] * len(df))
            if "comment" in file.name:
                reddit_comments_dataframes.append(df)
            else:
                reddit_submissions_dataframes.append(df)
        else:
            df.insert(3, "source", ["Twitter"] * len(df))
            twitter_dataframes.append(df)

    return twitter_dataframes, reddit_submissions_dataframes, reddit_comments_dataframes


def add_language_column(df: pd.DataFrame, content_column_names: list[str]):
    def check_language(post):
        text_content = post["combined_content"]
        # since it is only relevant whether it is in english or not other languages are ignored for now
        detected_language = "english" if detect_language(text_content) == "en" else "other"
        # detected_language = "english" if detect_contains_english(text_content) else "other"
        return detected_language

    # combine all the text column first (e.g. title and content for reddit submissions) into a new column with one
    # text divided by newline
    df["combined_content"] = df[[*content_column_names]].fillna('').apply("\n".join, axis=1)

    # add a new column to the given dataframe with the detected language of the text content
    lang_result = df.apply(lambda row: check_language(row), axis=1)
    df.insert(4, "detected_language", lang_result.reset_index(drop=True))
    # df.drop(columns=["combined_content"], axis=1, inplace=True)  # remove the temporarily created combined text column


def filter_post_time_period(df: pd.DataFrame, review_bombing_incident: str, is_reddit_submissions_file=False):
    # first, bring all date columns into the same format as the reviews have
    if is_reddit_submissions_file:
        df['created_at'] = pd.to_datetime(df['created_at'], unit="s")
        # df.drop(columns=['created_at_formatted'], axis=1, inplace=True)  # not needed anymore
    else:
        df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.strftime('%d.%m.%Y %H:%M:%S')

    start_time = review_bombing_incidents[review_bombing_incident]["social_media_start_time"]
    end_time = review_bombing_incidents[review_bombing_incident]["social_media_end_time"]
    # add new column with True / False based on whether the post was created during the time of the review bomb
    in_rb_period_result = df["created_at"].apply(
        lambda post_date: check_if_date_in_range(post_date, start_time, end_time))
    df.insert(6, "in_rb_time_period", in_rb_period_result.reset_index(drop=True))


def map_combine_social_media_data():
    # map each file in the tweets folder to its corresponding review bombing incident by adding the relevant
    # dataframe columns and combine all twitter files per rb incident afterwards
    print("Mapping social media files to corresponding review bombing incidents ...\n")

    for rb_name in review_bombing_incidents.keys():
        if rb_name in ["Crusader-Kings-II-Deus-Vult", "The-Long-Dark-GeForce-Now", "Superhot-VR"]:
            # no social media files were downloaded for these review bombing incidents as they are not used
            continue

        # create a subfolder for this review bombing incident if it doesn't exist yet
        Sub_Folder = OUTPUT_FOLDER_POSTS / rb_name
        if not Sub_Folder.is_dir():
            Sub_Folder.mkdir()
        else:
            print("WARNING: Subfolder already exists!")
            answer = input(f"Do you want to overwrite the existing folder for \"{rb_name}\"? [y/n]\n")
            if str.lower(answer) == "y" or str.lower(answer) == "yes":
                shutil.rmtree(Sub_Folder)
                Sub_Folder.mkdir()
            else:
                return

        twitter_df_list, reddit_submissions_df_list, reddit_comments_df_list = add_rb_information_to_social_media_files(rb_name)

        print("Combining files ...\n")
        twitter_df_combined = pd.concat(twitter_df_list)
        twitter_df_combined.sort_values(by="created_at", ascending=False, inplace=True)
        twitter_df_combined = twitter_df_combined.drop_duplicates(subset=['id']).reset_index(drop=True)

        # combine reddit comments and submissions into two different dataframes since they have different columns
        reddit_submissions_df_combined = pd.concat(reddit_submissions_df_list)
        reddit_submissions_df_combined.sort_values(by="created_at", ascending=False, inplace=True)
        reddit_submissions_df_combined = reddit_submissions_df_combined.drop_duplicates(subset=["id"]).reset_index(drop=True)

        reddit_comments_df_combined = pd.concat(reddit_comments_df_list)
        reddit_comments_df_combined.sort_values(by="created_at", ascending=False, inplace=True)
        reddit_comments_df_combined = reddit_comments_df_combined.drop_duplicates(subset=["id"]).reset_index(drop=True)

        # add a new column that flags each posts as either english or not to make it easier to separate these later
        print("Detecting language for each post ...")
        add_language_column(twitter_df_combined, content_column_names=["content"])
        add_language_column(reddit_submissions_df_combined, content_column_names=["title", "content"])
        add_language_column(reddit_comments_df_combined, content_column_names=["content"])

        # Check for each post if it's creation time is approximately in the period of the review bombing incident
        print("Filtering time range ...")
        filter_post_time_period(twitter_df_combined, rb_name)
        filter_post_time_period(reddit_submissions_df_combined, rb_name, is_reddit_submissions_file=True)
        filter_post_time_period(reddit_comments_df_combined, rb_name)

        twitter_df_combined.to_csv(Sub_Folder / f"twitter_combined_{rb_name}.csv", index=False)
        reddit_submissions_df_combined.to_csv(Sub_Folder / f"reddit_submissions_combined_{rb_name}.csv", index=False)
        reddit_comments_df_combined.to_csv(Sub_Folder / f"reddit_comments_combined_{rb_name}.csv", index=False)
        print(f"Finished with review bombing incident {rb_name}\n")


def cleanup_reddit_comments_for_submissions():
    # These were only fetched for certain review bombing incidents and rely on the cleaned reddit_submissions file
    # from the method above, so they need to be loaded and cleaned after the other social media files
    submission_comments_pattern = "comments_*submissions*"
    reddit_comment_files = [f for f in REDDIT_DATA_FOLDER.glob(submission_comments_pattern)]
    pprint.pprint(reddit_comment_files)

    for file in reddit_comment_files:
        print(f"Updating file {file} ...\n")
        df = pd.read_csv(file)
        incident_name = ""
        for rb_name in review_bombing_incidents.keys():
            # these reddit files contain the rb_name from the dictionary above in their file name (I know, just don't
            # touch anything related to it ...)
            if rb_name in file.stem:
                incident_name = rb_name
                affected_games = review_bombing_incidents[rb_name]["affected_games"]
                df.insert(3, "affected_games", [affected_games] * len(df))
                df.insert(3, "review_bombing_incident", [rb_name] * len(df))

        df.insert(3, "source", ["Reddit"] * len(df))
        print("Detecting language ...")
        add_language_column(df, content_column_names=["content"])
        print("Filtering time range ...")
        filter_post_time_period(df, incident_name)
        df.to_csv(OUTPUT_FOLDER_POSTS / incident_name / f"reddit_comments_for_submissions_{incident_name}.csv", index=False)


def combine_reddit_comments():
    for review_bombing_folder in OUTPUT_FOLDER_POSTS.iterdir():
        try:
            rb_name = review_bombing_folder.stem
            reddit_comments_df = pd.read_csv(review_bombing_folder / f"reddit_comments_combined_{rb_name}.csv")
            reddit_submission_comments_df = pd.read_csv(review_bombing_folder / f"reddit_comments_for_submissions_{rb_name}.csv")
            combined_comments_df = pd.concat([reddit_comments_df, reddit_submission_comments_df])
            combined_comments_df.sort_values(by="created_at", ascending=False, inplace=True)
            combined_comments_df = combined_comments_df.drop_duplicates(subset=['id']).reset_index(drop=True)

            combined_comments_df.to_csv(review_bombing_folder / f"combined_reddit_comments_{rb_name}.csv", index=False)
        except Exception as e:
            print(f"Could not load and combine two reddit comments files for folder {review_bombing_folder}: {e}\n")


if __name__ == "__main__":
    enable_max_pandas_display_size()

    if not OUTPUT_FOLDER_REVIEWS.is_dir():
        OUTPUT_FOLDER_REVIEWS.mkdir(parents=True)
    if not OUTPUT_FOLDER_POSTS.is_dir():
        OUTPUT_FOLDER_POSTS.mkdir(parents=True)

    add_game_name_columns = False
    if add_game_name_columns:
        add_game_name_to_game_files()

    add_review_bombing_information = False
    if add_review_bombing_information:
        # for rb_name in review_bombing_incidents:
        #     add_rb_information_to_game_files(rb_name)
        review_bombing_name = "Assassins-Creed-Unity"
        add_rb_information_to_game_files(review_bombing_name)

    map_social_media_data = False
    if map_social_media_data:
        map_combine_social_media_data()

    cleanup_combine_reddit_comments = False
    if cleanup_combine_reddit_comments:
        cleanup_reddit_comments_for_submissions()
        combine_reddit_comments()
