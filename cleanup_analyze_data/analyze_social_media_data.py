#!/usr/bin/python
# -*- coding: utf-8 -*-

import pprint
import pathlib
import datetime
import numpy as np
import pandas as pd
import ast
import re
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from cleanup_analyze_data.review_bombing_incidents_info import review_bombing_incidents
from sentiment_analysis_and_nlp.nlp_utils import apply_standard_text_preprocessing
from sentiment_analysis_and_nlp.sentiment_analysis import apply_sentiment_analysis
from sentiment_analysis_and_nlp.topic_modeling import apply_topic_modeling_social_media
from utils import utils


###########################################################################################


def analyze_authors_twitter(post_df: pd.DataFrame):
    author_df = post_df.iloc[:, 18:25]
    author_is_credible = False

    """
    Account-Erstelldatum, Expertise (Menge Reviews / Angesehen in Community / schon länger auf Plattform / ...),
    vorherige Reviews / Review Score Distribution, Anzahl geschriebener Reviews, Anzahl Spiele im Besitz, Anzahl Freunde
    """
    def is_author_credible(author_rows: pd.DataFrame):
        # similar to heuristics in data_labeling_test; check if author can be considered credible by different
        # conditions
        # "author_followers_count" > n, "author_friends_count" > n, "author_verified" == True and "author_created_at"
        # (account created recently during the rb_period or only shortly before post?)

        # low_follower_count = author_rows["author_followers_count"] < n
        # => not_credible if (low_follower_count or/and "author_verified" != True ...) else credible
        pass

    author_df["author_created_date"] = pd.to_datetime(author_df["author_created_at"], dayfirst=True,
                                                      format='mixed').dt.strftime('%d.%m.%Y')
    # compare "author_created_date" with "created_date" in posts_df

    author_df.groupby("author_id").apply(is_author_credible)
    return author_is_credible


def analyze_twitter_posts(twitter_posts_df: pd.DataFrame):
    rb_incident_key = twitter_posts_df.at[0, "review_bombing_incident"]
    social_media_info_dict["review_bombing_incident"].append(rb_incident_key)

    twitter_posts_df["created_date"] = pd.to_datetime(twitter_posts_df["created_at"], dayfirst=True,
                                                      format='mixed').dt.strftime('%d.%m.%Y')
    # tweet_per_date_counts = twitter_posts_df['created_date'].value_counts()
    # print(f"{tweet_per_date_counts}\n")

    twitter_posts_df["reactions_score"] = twitter_posts_df["like_count"] + twitter_posts_df["reply_count"] + twitter_posts_df["retweet_count"]

    ############################################
    # used_emojis column: either only check how many (i.e. len()) emojis per tweet or try to get sentiment from them ?
    # use a converter to convert saved list back to a list (by default it would be a string)
    try:
        emoji_count_list: pd.Series = twitter_posts_df['used_emojis'].apply(lambda x: len(ast.literal_eval(x)))
        emoji_count_list_values = emoji_count_list.value_counts()
        print(emoji_count_list_values)
        twitter_posts_df['used_emojis_count'] = emoji_count_list
    except Exception as e:
        print(e)

    emoji_series_cleaned = twitter_posts_df["content"].str.findall(r":\w+:")
    emoji_series_cleaned = emoji_series_cleaned.str.join(' ')
    twitter_posts_df['used_emojis_cleaned'] = emoji_series_cleaned.str.replace(":", "")

    ############################################
    hashtags_series = twitter_posts_df["hashtags"].str.findall(r"text': '(\w+)'}")   # use capturing group (...)
    twitter_posts_df['hashtags_cleaned'] = hashtags_series.str.join(" ")

    ############################################
    english_twitter_posts = twitter_posts_df[twitter_posts_df["detected_language"] == "english"]
    non_english_twitter_posts = twitter_posts_df[twitter_posts_df["detected_language"] == "other"]
    twitter_posts_rb_time = twitter_posts_df[twitter_posts_df["in_rb_time_period"]]
    twitter_posts_not_rb_time = twitter_posts_df[~twitter_posts_df.index.isin(twitter_posts_rb_time.index)]

    ########### find duplicates during and outside of the review bombing time period ###########
    duplicates_rb_time = twitter_posts_rb_time.loc[twitter_posts_rb_time.duplicated(subset=["content"], keep=False), :]
    duplicate_percentage_rb_time = (len(duplicates_rb_time) / len(twitter_posts_rb_time)) * 100
    print(f"\n{len(duplicates_rb_time)} / {len(twitter_posts_rb_time)} "
          f"({duplicate_percentage_rb_time:.2f} %) duplicate posts during the time period of the review bombing:")
    # pprint.pprint(duplicates_rb_time)

    duplicates_not_rb_time = twitter_posts_not_rb_time.loc[
                             twitter_posts_not_rb_time.duplicated(subset=["content"], keep=False), :]
    duplicate_percentage_not_rb_time = (len(duplicates_not_rb_time) / len(twitter_posts_not_rb_time)) * 100
    print(f"\n{len(duplicates_not_rb_time)} / {len(twitter_posts_not_rb_time)} "
          f"({duplicate_percentage_not_rb_time:.2f} %) duplicate posts outside the time period of the review bombing")

    # don't remove duplicates because they contain some value in this case (?)
    # rb_time_posts_no_duplicates = twitter_posts_rb_time.drop_duplicates(subset=["content"])
    # not_rb_time_posts_no_duplicates = twitter_posts_not_rb_time.drop_duplicates(subset=["content"])

    ############################################
    # analyze the post authors
    # analyze_authors_twitter(twitter_posts_df)
    # analyze_authors(twitter_posts_rb_time)

    ############################################
    # subset_df = twitter_posts_df[["content", "hashtags_cleaned", "used_emojis_cleaned", "used_emojis_count"]]

    # "review_bomb_type" in review_bombing_incidents dict mit sentiment ergebnis vergleichen ?
    # -> negative Tweets / Average Sentiment für negatives RB ?
    rb_type = review_bombing_incidents[rb_incident_key]["review_bomb_type"]

    # TODO drop duplicates before topic modeling and sentiment analysis ?

    # only on english data and only in rb time period
    english_in_rb_time_df = twitter_posts_df[(twitter_posts_df["in_rb_time_period"]) & (twitter_posts_df["detected_language"] == "english")]

    apply_sentiment_analysis(english_in_rb_time_df, "combined_content", col_for_sentiment_analysis="text_cleaned",
                             perform_preprocessing=False, social_media_data=True)
    avg_sentiment_for_incident = english_in_rb_time_df['sentiment_score_sentence_level'].mean()
    social_media_info_dict["avg_sentiment_rb_period - Twitter"].append(avg_sentiment_for_incident)

    should_perform_topic_modeling = True
    if should_perform_topic_modeling:
        # topic modeling only on english data
        # TODO use combined_content instead of text_cleaned ?
        topics_df, combined_topics = apply_topic_modeling_social_media(english_twitter_posts, rb_name,
                                                                       "combined_content",
                                                                       col_for_modeling="text_cleaned",
                                                                       perform_preprocessing=False,
                                                                       social_media_data=True)

        social_media_info_dict["Topic 0 - Twitter"].append(combined_topics[0] if 1 < len(combined_topics) else "")
        social_media_info_dict["Topic 1 - Twitter"].append(combined_topics[1] if 2 < len(combined_topics) else "")
        social_media_info_dict["Topic 2 - Twitter"].append(combined_topics[2] if 3 < len(combined_topics) else "")

    #  plot sentiment together with the graph above to find out if sentiment in rb time changed
    #  -> maybe this helps in identifying start and end date?
    ############################################
    # show the number of tweets per date to get an idea of the daily activity for this topic / incident
    tweet_counts = twitter_posts_rb_time.groupby('created_date').size()
    # convert string date in index to actual date so it can be sorted for the plot
    tweet_counts.index = pd.to_datetime(tweet_counts.index, dayfirst=True)  # .sort_index()

    plt.figure(figsize=(10, 5))
    plt.bar(tweet_counts.index, tweet_counts.values, align="center")
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.title(f'Twitter Activity for {rb_incident_key} during time of incident')
    plt.xticks(rotation=0)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    plt.savefig(OUTPUT_FOLDER / f"twitter_activity_{rb_incident_key}.svg", format="svg")

    daily_sentiment_scores = list()

    def calculate_daily_sentiment(day_df: pd.DataFrame):
        day = day_df.name
        avg_sentiment_for_day = day_df["sentiment_score_sentence_level"].mean()
        daily_sentiment_dict = {
            "Date": day,
            "Avg Daily Sentiment": avg_sentiment_for_day,
        }
        daily_sentiment_scores.append(daily_sentiment_dict)

    english_in_rb_time_df.groupby("created_date").apply(calculate_daily_sentiment)
    daily_sentiment_df = pd.DataFrame(daily_sentiment_scores)
    # convert to datetime to sort the dataframe
    daily_sentiment_df["Datetime"] = pd.to_datetime(daily_sentiment_df["Date"], dayfirst=True)
    daily_sentiment_df = daily_sentiment_df.sort_values(by='Datetime')

    plt.figure(figsize=(10, 5))
    plt.plot(daily_sentiment_df["Datetime"], daily_sentiment_df["Avg Daily Sentiment"])
    plt.xlabel('Date')
    plt.ylabel('Avg Daily Sentiment')
    plt.title(f'Average daily sentiment for {rb_incident_key} during time of incident')
    plt.xticks(rotation=10)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    plt.savefig(OUTPUT_FOLDER / f"daily_sentiment_{rb_incident_key}.svg", format="svg")

    print("breakpoint")


def analyze_reddit_submissions(reddit_submission_df: pd.DataFrame):
    """
    groupby(by=["subreddit"]) and / or sort by ["subreddit", "created_at"] ?

    "score" for result of number of upvotes
    "num_comments"
    -> use "combined_content" column here instead of "content"
    """
    # TODO create plot here too as for twitter but maybe combine both reddit comments and submissions into one plot?
    pass


def analyze_reddit_comments(reddit_comment_df: pd.DataFrame):
    """
    groupby(by=["subreddit"]) to get comments per subreddit?
        -> or sort by ["subreddit", "created_at"] to sort per date in subreddit ?

    groupby(by=["original_post_date", "original_post_author_id"]) to get all comments for one submission

    "upvote_score": upvotes - downvotes (can be negative)
    """
    # TODO for reddit topic modeling: maybe combine all comments text with the original submission text where possible ?
    #  -> find original submission by merge at (original_post_date (+original_post_author_id))
    pass


if __name__ == "__main__":
    utils.enable_max_pandas_display_size()
    DATA_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "posts"
    OUTPUT_FOLDER = pathlib.Path(__file__).parent / "analysis_results"

    if not OUTPUT_FOLDER.is_dir():
        OUTPUT_FOLDER.mkdir()

    rb_incidents = ["Skyrim-Paid-Mods", "Assassins-Creed-Unity", "Firewatch", "Mortal-Kombat-11",
                    "Borderlands-Epic-Exclusivity", "Ukraine-Russia-Conflict"]
    # rb_incidents = ["Assassins-Creed-Unity", "Skyrim-Paid-Mods"]

    social_media_info_dict = {"review_bombing_incident": [], "avg_sentiment_rb_period - Twitter": [],
                              "Topic 0 - Twitter": [], "Topic 1 - Twitter": [], "Topic 2 - Twitter": []}

    for rb_name in rb_incidents:
        incident_folder = DATA_FOLDER / rb_name
        twitter_data = pd.read_csv(incident_folder / f"twitter_combined_{rb_name}.csv")
        # TODO Reddit
        # reddit_submission_data = pd.read_csv(incident_folder / f"reddit_submissions_combined_{rb_name}.csv")
        # reddit_comment_data = pd.read_csv(incident_folder / f"combined_reddit_comments_{rb_name}.csv")

        apply_standard_text_preprocessing(twitter_data, text_col="combined_content", remove_punctuation=False,
                                          remove_stopwords=False, is_social_media_data=True)

        analyze_twitter_posts(twitter_data)
        # analyze_reddit_submissions(reddit_submission_data)
        # analyze_reddit_comments(reddit_comment_data)

    # combine everything into one dataframe for Twitter and Reddit
    social_media_info_df = pd.DataFrame.from_dict(social_media_info_dict)

    # add to reviews for classification
    dataframe = pd.read_csv(pathlib.Path(__file__).parent.parent / "final_annotation_all_projects_review_analysis.csv")
    df_with_social_media_data = dataframe.merge(social_media_info_df, how="inner")
    df_with_social_media_data.to_csv(pathlib.Path(__file__).parent.parent / 
                                     "final_annotation_all_projects_review_social_media_analysis.csv", index=False)

