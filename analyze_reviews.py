#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import pprint
import pathlib
from datetime import datetime
import pandas as pd
from sentiment_analysis_and_nlp.nlp_utils import split_into_sentences
from useful_code_from_other_projects import utils
from useful_code_from_other_projects.utils import enable_max_pandas_display_size
from profanity_check import predict, predict_prob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import textstat
from textblob import TextBlob, Word

DATA_FOLDER = pathlib.Path(__file__).parent / "data_for_analysis"


def analyze_steam_review_graph(review_graph_data: dict):
    # convert unix timestamps to readable dates
    start_date_unix = review_graph_data["start_date"]
    end_date_unix = review_graph_data["end_date"]
    start_date = datetime.fromtimestamp(start_date_unix).strftime('%d.%m.%Y')
    end_date = datetime.fromtimestamp(end_date_unix).strftime('%d.%m.%Y')
    print(f"\nSteam Review Graph Data from {start_date = } to {end_date = }")

    steam_review_graph_dict = {"month": [], "recommendations_up": [], "recommendations_down": []}
    months_data = review_graph_data["review_data"]
    for month in months_data:
        date_utc = month["date"]
        date = datetime.fromtimestamp(date_utc).strftime('%m.%Y')
        upvotes = month["recommendations_up"]
        downvotes = month["recommendations_down"]
        print(f"{date}: {upvotes} Upvotes / {downvotes} Downvotes")
        steam_review_graph_dict["month"].append(date)
        steam_review_graph_dict["recommendations_up"].append(upvotes)
        steam_review_graph_dict["recommendations_down"].append(downvotes)

    # convert data to csv
    review_graph_df = pd.DataFrame(steam_review_graph_dict)
    review_graph_df.to_csv("steam_review_graph_cyberpunk_2077.csv", index=False)

    # TODO search for temporal bursts and other suspicious changes in the number of reviews per month


def analyze_reviewer(reviewer: pd.Series):
    # TODO analyze combined with review !! look for the same username in metacritic & steam?
    #  -> Auffälligkeiten: erst kurz vor review date erstellt, sehr wenige (< 3) reviews, average score sehr niedrig,
    #  steam_playtime_at_review sehr kurz, real_name (steam) bekannt oder nicht? bzw. an sich ausführlicheres Profil
    #  oder nicht (vgl. num_owned_games, num_friends, profil_private), ...
    pass


def check_profanity(text: str):
    contains_profanity = predict([text])
    profanity_probability = predict_prob([text])
    print(f"The text \"{text}\"\ncontains profanity: {contains_profanity} (with {profanity_probability[0] * 100} %)")


def check_spelling_errors(text: str, correct_spelling=False):
    blob = TextBlob(text)
    num_spelling_errors = 0
    w: Word
    for w in blob.words:
        print(w.spellcheck())  # word_suggestion, confidence
    print(f"Number of spelling errors in text: {num_spelling_errors}")

    if correct_spelling:
        corrected_blob = blob.correct()
        return str(corrected_blob)
    return text


def calculate_readability(text: str):
    # see https://www.youtube.com/watch?v=abgggvnrGBg &
    # https://python.plainenglish.io/calculate-statistical-features-for-readability-from-text-using-python-5ace1da7eb7b)

    # textstat.set_lang("en_US")

    # Important: readability scores can be misleading in the case of short text content !
    # TODO in der Literatur (vgl. Investigating Helpfulness of Video Game Reviews on the Steam Platform) werden
    #  deshalb alle Reviews mit weniger als 100 (!) Wörtern entfernt  => macht das Sinn? dann bleiben nur noch wenige...
    fre = textstat.flesch_reading_ease(text)
    fkg = textstat.flesch_kincaid_grade(text)
    gf = textstat.gunning_fog(text)
    si = textstat.smog_index(text)
    ari = textstat.automated_readability_index(text)
    cli = textstat.coleman_liau_index(text)
    lwf = textstat.linsear_write_formula(text)
    # TODO how to combine all readability scores? taking average probably not good ? combine into one feature vector?
    print(f"\nReadability Scores:\n"
          f"Flesch-Reading-Ease: {fre}\nFlesch-Kincaid-Grade: {fkg}\nGunning-Fog: {gf}\nSmog-Index: {si}\n"
          f"Automated-Readability-Index: {ari}\nColeman-Liau-Index: {cli}\nLinsear-Write-Formula: {lwf}\n")

    difficult_words = textstat.difficult_words_list(text)
    char_count = textstat.char_count(text, ignore_spaces=True)
    word_count = textstat.lexicon_count(text, removepunct=True)  # number of words
    sentence_count = textstat.sentence_count(text)
    avg_sentence_len = textstat.avg_sentence_length(text)  # combination of the two above
    print(f"Number of difficult words: {len(difficult_words)}\nCharacter Count: {char_count}\nWord Count: "
          f"{word_count}\nSentence Count: {sentence_count}\nAverage Sentence Length: {avg_sentence_len}\n")


def calculate_cosine_similarity(text1: str, text2: str):
    documents = [text1, text2]
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)

    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(
        doc_term_matrix,
        columns=count_vectorizer.get_feature_names_out(),
        index=["text1", "text2"],
    )
    print(df)
    print(cosine_similarity(df, df))

    # alternatively use tf-idf instead of count:
    # see https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
    tfidf = TfidfVectorizer(min_df=1).fit_transform(documents)
    pairwise_similarity = tfidf * tfidf.T
    print(pairwise_similarity)
    return pairwise_similarity


"""
def create_similarity_heatmap(cosine_similarity, cmap="YlGnBu"):
    df = pd.DataFrame(cosine_similarity)
    df.columns = labels    # labels = [headline[:20] for headline in headlines]
    df.index = labels
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(df, cmap=cmap)
"""


def calculate_text_similarity(text1: str, text2: str, spacy_nlp):
    """
    See https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python for a good overview.
    """
    # convert text to word embeddings with spaCy
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)
    embeddings1 = [spacy_nlp(sentence).vector for sentence in sentences1]
    embeddings2 = [spacy_nlp(sentence).vector for sentence in sentences2]
    # alternatively use Gensim to use Doc2Vec for sentence embeddings or Elmo for contextual embeddings

    # calculate cosine similarity
    # cosine_similarity = calculate_cosine_similarity(text1, text2)
    # create_similarity_heatmap(cosine_similarity)


def find_duplicate_or_similar_reviews():
    # use something like text similarity to find near duplicate review texts from different users
    pass


def analyze_review(review: pd.Series):
    # TODO calculate all necessary features for this review and return new pd.Series
    #  with the calculated features (cosine similarity with other reviews?, profanity, length, readabiliy)
    print(f"Analyzing new review ...\n")
    game_description = review["game_description"]
    review_text = review["review"]
    check_profanity(review_text)
    check_spelling_errors(review_text)
    calculate_readability(review_text)
    # calculate_text_similarity(review_text, game_description)

    # analyze_reviewer()


def combine_metacritic_steam_reviews(reviews_steam: pd.DataFrame, reviews_metacritic: pd.DataFrame,
                                     game_info_steam: pd.DataFrame, game_info_metacritic: pd.DataFrame):
    """
    Combine metacritic and steam reviews into one unified dataframe. As steam reviews have far more features the
    Metacritic rows will contain a lot of empty values.
    """
    # pd.options.mode.chained_assignment = None  # disable some warnings
    # add a common column to the dataframes for merging and to differentiate between the review platforms
    merge_column = "source"
    reviews_steam.insert(0, merge_column, ["Steam"] * len(reviews_steam))
    reviews_metacritic.insert(0, merge_column, ["Metacritic"] * len(reviews_metacritic))
    game_info_steam.insert(0, merge_column, ["Steam"] * len(game_info_steam))
    game_info_metacritic.insert(0, merge_column, ["Metacritic"] * len(game_info_metacritic))

    # rename columns so the same content from both dataframes ends up in the same column
    reviews_steam = reviews_steam.rename(
        columns={"content": "review", "created_at_formatted": "review_date", "useful_score": "helpful_votes",
                 "author_num_reviews": "author_num_game_reviews", "author_username": "username"})
    game_info_steam = game_info_steam.rename(columns={"short_description": "game_description"})

    merged_reviews_df = pd.concat([reviews_steam, reviews_metacritic], axis=0, ignore_index=True)
    merged_game_info_df = pd.concat([game_info_steam, game_info_metacritic], axis=0, ignore_index=True)
    # merge the reviews and the general info for the game
    combined_df = merged_reviews_df.merge(merged_game_info_df, on=[merge_column], how='outer')

    # drop unnecessary columns
    combined_df = combined_df.drop(columns=["review_id", "rating_positive", "created_at", "last_updated", "author_id",
                                            "comment_count", "platform", "profile_visibility", "author_country_code",
                                            "profile_url", "game_id", "title", "game_title", "price_euro"],
                                   axis=1)
    combined_df.insert(0, "Game", ["Cyberpunk 2077"] * len(combined_df))  # add the game name as a new column
    return combined_df


def prepare_data():
    # load metacritic and steam reviews
    steam_review_data = pd.read_csv(DATA_FOLDER / "steam" / "steam_user_reviews_Cyberpunk 2077_12_2020_06_2023.csv")
    steam_general_game_info = pd.read_csv(DATA_FOLDER / "steam" / "steam_general_info_Cyberpunk_2077.csv")
    metacritic_review_data = pd.read_csv(
        DATA_FOLDER / "metacritic" / "filtered_metacritic_user_reviews_pc_cyberpunk-2077.csv")
    metacritic_game_info = pd.read_csv(DATA_FOLDER / "metacritic" / "game_info_pc_cyberpunk-2077.csv")

    # manually select some reviews for testing:
    test_reviews = ["dudejustin", "Marko156", "oslo1", "Cyberbug2020", "Alpha_Octav", "Ariegan", "Brzytwa",
                    "spencieboi", "BahamutxD", "TheGamer98"]
    filtered_metacritic = metacritic_review_data.loc[metacritic_review_data["username"].isin(test_reviews)]
    print(filtered_metacritic)
    print("\n######################\n")

    test_steam_reviews = [139928616, 140225020, 140101316, 83758053, 83248147, 82899690, 81940229, 81948825,
                          81950472, 81949288, 81958813]
    steam_review_data = utils.remove_linebreaks_from_pd_cells(steam_review_data, column_name="content")
    filtered_steam = steam_review_data.loc[steam_review_data["review_id"].isin(test_steam_reviews)]
    print(filtered_steam)
    print("\n######################\n")

    combined_review_df = combine_metacritic_steam_reviews(filtered_steam, filtered_metacritic,
                                                          steam_general_game_info, metacritic_game_info)
    combined_review_df.to_csv("combined_steam_metacritic_df.csv", index=False)


if __name__ == "__main__":
    enable_max_pandas_display_size()
    # prepare_data()

    """
    with open(DATA_FOLDER / "steam" / "steam_reviews_timeseries_Cyberpunk_2077.json", "r") as f:
        steam_graph_data = json.load(f)
    analyze_steam_review_graph(steam_graph_data)
    """

    review_df = pd.read_csv("combined_steam_metacritic_df.csv")
    review_df.apply(lambda row: analyze_review(row), axis=1)
    # review_df["profanity_in_percent"] = predict_prob(review_df["review"]) * 100
    # print(review_df)
