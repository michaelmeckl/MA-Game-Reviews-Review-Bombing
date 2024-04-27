#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
from datetime import datetime
import numpy as np
import pandas as pd
from spellchecker import SpellChecker
from sentiment_analysis_and_nlp import nltk_utils
from sentiment_analysis_and_nlp.nlp_utils import apply_standard_text_preprocessing
from sentiment_analysis_and_nlp.sentiment_analysis import apply_sentiment_analysis
from sentiment_analysis_and_nlp.spacy_utils import SpacyUtils
from sentiment_analysis_and_nlp.topic_modeling import apply_topic_modeling_reviews
from utils import utils
from profanity_check import predict, predict_prob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import textstat
from textblob import TextBlob, Word

spell = SpellChecker()


def analyze_reviewers(df: pd.DataFrame):
    average_num_ratings = np.nanmean(df["author_ratings_overall"])
    average_num_reviews_overall = np.nanmean(df["author_reviews_overall"])
    average_num_reviews = np.nanmean(df["author_num_game_reviews"])

    average_num_games = np.nanmean(df["author_num_owned_games"])
    average_num_friends = np.nanmean(df["author_num_friends"])
    average_steam_level = np.nanmean(df["author_steam_level"])

    def calc_author_credibility_experience(row: pd.DataFrame):
        """
        Steam Columns: author_playtime_overall_min, author_playtime_at_review_min, username, author_steam_level,
                       profile_created, profile_visibility, author_num_friends, author_num_owned_games
        Metacritic Columns: author_ratings_overall, author_reviews_overall, author_average_score
        Both: author_num_game_reviews
        """
        experience_score = 0.0
        credibility_score = 0.0

        if row["author_num_game_reviews"] > average_num_reviews:
            credibility_score += 1
            experience_score += 1
        elif row["author_num_game_reviews"] < 2:   # if the author has no written other reviews
            credibility_score -= 1
            experience_score -= 1

        review_date = row["review_date"]

        # based on the platform we need different columns for the author
        review_is_from_steam = True if row["source"] == "Steam" else False
        if review_is_from_steam:
            # refund time on Steam is below 2 hours, so everything above can be considered pretty credible
            if row["author_playtime_at_review_min"] >= 120:
                credibility_score += 1

            # author continued playing the game after leaving the review  -> credible
            if row["author_playtime_overall_min"] > row["author_playtime_at_review_min"]:
                credibility_score += 1

            # author account created shortly before review was posted ? -> not credible
            account_creation_date = row["profile_created"]
            if not pd.isnull(account_creation_date):
                review_datetime = datetime.strptime(review_date, '%d.%m.%Y %H:%M:%S')
                account_datetime = datetime.strptime(account_creation_date, '%d.%m.%Y %H:%M:%S')
                delta = review_datetime - account_datetime
                num_days_between = delta.days

                if num_days_between >= 2:
                    credibility_score += 1
                    experience_score += 1
                else:
                    credibility_score -= 1
                    experience_score -= 1

            # author_num_friends, author_num_owned_games, author_steam_level  greater than average  -> credible
            # if these are 0 or below 2  -> not credible
            if row["author_steam_level"] > average_steam_level:
                credibility_score += 1
                experience_score += 1
            elif row["author_steam_level"] < 2:
                credibility_score -= 1
                experience_score -= 1

            if row["author_num_friends"] > average_num_friends:
                credibility_score += 1
            elif row["author_num_friends"] == 0:
                credibility_score -= 1

            if row["author_num_owned_games"] > average_num_games:
                credibility_score += 1
            elif row["author_num_owned_games"] < 2:    # author owns only this one game
                credibility_score -= 1
        else:
            # check if author has more ratings than the average author -> credible
            if row["author_ratings_overall"] > average_num_ratings:
                credibility_score += 1
                experience_score += 1
            elif row["author_ratings_overall"] < 2:
                credibility_score -= 1
                experience_score -= 1

            if row["author_reviews_overall"] > average_num_reviews_overall:
                credibility_score += 1
                experience_score += 1
            elif row["author_reviews_overall"] < 2:
                credibility_score -= 1
                experience_score -= 1

            # if average is extreme  -> not credible
            if row["author_average_score"] > 8:
                credibility_score -= 1
            if row["author_average_score"] < 4:
                credibility_score -= 1

        # no negative credibility or experience, just use 0 in that case
        credibility_score = credibility_score if credibility_score > 0.0 else 0.0
        experience_score = experience_score if experience_score > 0.0 else 0.0
        return credibility_score, experience_score

    # check if author can be considered credible or experienced by using different conditions
    df[["author_credibility", "author_experience"]] = df.apply(lambda row: pd.Series(calc_author_credibility_experience(row)), axis=1)


def check_profanity(text: str):
    contains_profanity_label = predict([text])
    # profanity_probability = predict_prob([text])
    # print(f"The text \"{text}\"\ncontains profanity: {contains_profanity_label} (with {profanity_probability[0] * 100} %)")
    contains_profanity = True if contains_profanity_label[0] == 1 else False
    return contains_profanity


def check_spelling(text: str, correct_spelling=False):
    blob = TextBlob(text)
    num_spelling_errors = 0
    w: Word
    for w in blob.words:
        print(w.spellcheck())  # word_suggestion, confidence
    print(f"Number of spelling errors in text: {num_spelling_errors}")

    if correct_spelling:
        corrected_blob = blob.correct()
        return str(corrected_blob)
    return num_spelling_errors


def get_num_spelling_errors(text: str):
    tokenized_text = nltk_utils.tokenize_text(text, is_social_media_data=False)
    # find those words that may be misspelled
    misspelled = spell.unknown(tokenized_text)
    return len(misspelled)


def calculate_readability(text: str):
    # see https://www.youtube.com/watch?v=abgggvnrGBg &
    # https://python.plainenglish.io/calculate-statistical-features-for-readability-from-text-using-python-5ace1da7eb7b)

    # textstat.set_lang("en_US")

    # Important: readability scores can be misleading in the case of short text content !
    # TODO in der Literatur (vgl. Investigating Helpfulness of Video Game Reviews on the Steam Platform) werden
    #  deshalb alle Reviews mit weniger als 100 (!) Wörtern entfernt  => macht das Sinn? dann bleiben nur noch wenige...
    fre = textstat.flesch_reading_ease(text)
    """
    fkg = textstat.flesch_kincaid_grade(text)
    gf = textstat.gunning_fog(text)
    si = textstat.smog_index(text)
    ari = textstat.automated_readability_index(text)
    cli = textstat.coleman_liau_index(text)
    lwf = textstat.linsear_write_formula(text)
    #  how to combine all readability scores?
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
    """
    # see https://github.com/textstat/textstat for these values
    if fre >= 90:
        return "Very Easy"
    elif fre >= 70:
        return "Easy"
    elif fre >= 60:
        return "Standard"
    elif fre >= 30:
        return "Difficult"
    else:
        return "Very Difficult"


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


def calculate_text_similarity(text1: str, text2: str, spacy_nlp):
    """
    See https://www.newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python for a good overview.
    """
    # convert text to word embeddings with spaCy
    spacy_utils = SpacyUtils()
    sentences1 = spacy_utils.split_into_sentences(text1)
    sentences2 = spacy_utils.split_into_sentences(text2)
    embeddings1 = [spacy_nlp(sentence).vector for sentence in sentences1]
    embeddings2 = [spacy_nlp(sentence).vector for sentence in sentences2]
    # alternatively use Gensim to use Doc2Vec for sentence embeddings or Elmo for contextual embeddings

    # calculate cosine similarity
    # cosine_similarity = calculate_cosine_similarity(text1, text2)
    # create_similarity_heatmap(cosine_similarity)


def analyze_review(review: pd.Series):
    # calculate all necessary features for this review and return new pd.Series
    #  with the calculated features (cosine similarity with other reviews?, profanity, length, readabiliy)
    print(f"Analyzing new review ...\n")
    game_description = review["game_description"]
    review_text = review["review"]
    check_profanity(review_text)
    check_spelling(review_text)
    calculate_readability(review_text)
    # calculate_text_similarity(review_text, game_description)


def start_review_analysis(dataframe: pd.DataFrame):
    """
    dataframe['readability_flesch_reading'] = dataframe[text_column].apply(lambda x: calculate_readability(x))
    dataframe['num_spelling_errors'] = dataframe[text_column].apply(lambda x: get_num_spelling_errors(x))
    dataframe['contains_profanity'] = dataframe[text_column].apply(lambda x: check_profanity(x))
    analyze_reviewers(dataframe)

    apply_sentiment_analysis(dataframe, text_column, col_for_sentiment_analysis="text_cleaned", perform_preprocessing=False,
                             social_media_data=False)
    """

    # add columns for each incident with the top n topics for this incident
    topics_per_incident_list = list()

    def get_topics_for_incident(incident_df: pd.DataFrame):
        rb_name = incident_df.name
        topics_df, combined_topics = apply_topic_modeling_reviews(incident_df, rb_name, "review",
                                                                  col_for_modeling="text_cleaned",
                                                                  perform_preprocessing=False,
                                                                  social_media_data=False)
        topics_per_incident_info = {
            "project": rb_name,
            "Topic 0": combined_topics[0] if 1 < len(combined_topics) else "",
            "Topic 1": combined_topics[1] if 2 < len(combined_topics) else "",
            "Topic 2": combined_topics[2] if 3 < len(combined_topics) else "",
            "Topic 3": combined_topics[3] if 4 < len(combined_topics) else "",
            "Topic 4": combined_topics[4] if 5 < len(combined_topics) else "",
        }
        topics_per_incident_list.append(topics_per_incident_info)

    dataframe.groupby("project").apply(get_topics_for_incident)

    topics_per_incident_df = pd.DataFrame(topics_per_incident_list)
    df_with_topics = dataframe.merge(topics_per_incident_df, how="inner")
    print("debug")

    # save as a new csv for now; probably saver than overwriting the existing one
    df_with_topics.to_csv(pathlib.Path(__file__).parent.parent / "final_annotation_all_projects_review_analysis.csv",
                          index=False)


if __name__ == "__main__":
    utils.enable_max_pandas_display_size()
    DATA_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "reviews"

    # load the final annotated data
    combined_annotated_data = pd.read_csv(pathlib.Path(__file__).parent.parent /
                                          "combined_final_annotation_all_projects_updated.csv")
    apply_standard_text_preprocessing(combined_annotated_data, text_col="review", remove_stopwords=False,
                                      remove_punctuation=False)

    start_review_analysis(combined_annotated_data)
