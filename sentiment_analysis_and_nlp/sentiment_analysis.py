#!/usr/bin/python
# -*- coding: utf-8 -*-
import pathlib
import pprint
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize, FreqDist
from textblob import TextBlob
from sentiment_analysis_and_nlp.nlp_utils import apply_standard_text_preprocessing


def classify_sentiment_score(score: float, result_as_number=True):
    """
    Use like this after the function above:
     df["sentiment_score_vader"] = pd.to_numeric(df["sentiment_score_vader"], errors="coerce")
     df["sentiment_classification_vader"] = df["sentiment_score_vader"].apply(classify_sentiment_score)
    """
    if score > 0.2:
        class_result = "positive", 1
    elif score < -0.2:
        class_result = "negative", -1
    else:
        class_result = "neutral", 0
    # return class_result[1] if result_as_number else class_result[0]
    return class_result


def textblob_sentiment_analysis(df, df_text_col):
    def get_textblob_sentiment(post):
        blob = TextBlob(post)
        # TODO calc at sent level too ? -> for now only calc at whole text level
        # print(blob.sentences)  # split into sentences
        # lemmata = [word.lemmatize() for word in blob.words]
        text_sentiment = blob.sentiment
        # The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range
        # [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
        # print(text_sentiment)
        return text_sentiment.polarity

    df['sentiment_score_textblob'] = df[df_text_col].apply(lambda x: get_textblob_sentiment(x))
    # create a new column to classify the polarity scores
    df[["label_text_textblob", "label_textblob"]] = df["sentiment_score_textblob"].apply(
        lambda score: pd.Series(classify_sentiment_score(score)))

    sentiment_mean = df['sentiment_score_textblob'].mean()
    print(f"Mean sentiment score for all texts: {sentiment_mean:.2f}")
    print(df["label_textblob"].value_counts())
    print(df["label_textblob"].value_counts(normalize=True) * 100)  # show label distribution (i.e. as percentage)


def vader_sentiment_analysis(df, df_text_col):
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_score(post):
        sentiment = analyzer.polarity_scores(post)
        return sentiment['compound']

    df['sentiment_score'] = df[df_text_col].apply(lambda x: get_sentiment_score(x))
    # create a new column to classify the polarity scores
    df[["label_text_whole", "label_whole"]] = df["sentiment_score"].apply(lambda score: pd.Series(classify_sentiment_score(
        score)))

    sentiment_mean = df['sentiment_score'].mean()
    print(f"Mean sentiment score for all texts: {sentiment_mean:.2f}")
    print(df["label_whole"].value_counts())
    print(df["label_whole"].value_counts(normalize=True) * 100)  # show label distribution (i.e. as percentage)

    # Plot a histogram of sentiment scores
    plt.hist(df['sentiment_score'], bins=20, color='blue', alpha=0.5)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Vader Sentiment Scores')
    plt.show()


def vader_sentiment_analysis_sentence_level(df: pd.DataFrame, df_text_col, debug=False):
    sia = SentimentIntensityAnalyzer()
    """
    results = []
    # split content into sentences first so the polarity_scores function works
    sentence_dict = [(text, tokenize.sent_tokenize(text)) for text in text_list]

    for text, sentences in sentence_dict:
        polarity_scores = []
        for sentence in sentences:
            polarity_score = sia.polarity_scores(sentence)
            if debug:
                print(f"\nSentence: \n{sentence}")
                for field in sorted(polarity_score):
                    print('{0}: {1}, '.format(field, polarity_score[field]), end='')
            polarity_scores.append(polarity_score["compound"])

        mean_polarity_score = np.mean(polarity_scores)
        results.append((text, mean_polarity_score))

    # use from_records() as we have a list of dictionaries
    sentiment_df = pd.DataFrame.from_records(results, columns=['text', 'sentiment_score_sentence_level'])
    """
    
    def calculate_sentence_sentiment(post_text):
        sentences = tokenize.sent_tokenize(post_text)
        polarity_scores = []
        for sentence in sentences:
            polarity_score = sia.polarity_scores(sentence)
            polarity_scores.append(polarity_score["compound"])
        
        mean_polarity_score = np.mean(polarity_scores)
        return mean_polarity_score

    df['sentiment_score_sentence_level'] = df[df_text_col].apply(lambda x: calculate_sentence_sentiment(x))
    print("\n ############################################################### \n")

    # create a new column to classify the polarity scores
    df[["label_text", "label"]] = df["sentiment_score_sentence_level"].apply(lambda score: pd.Series(classify_sentiment_score(score)))

    sentiment_mean = df['sentiment_score_sentence_level'].mean()
    print(f"Mean sentiment score for all texts: {sentiment_mean:.2f}")
    print(df.label.value_counts())
    print(df.label.value_counts(normalize=True) * 100)  # show label distribution (i.e. as percentage)

    # show some sentiment analysis results
    if debug:
        print("\nSome positive posts:")
        pprint.pprint(list(df[df["label"] == 1]["combined_content"])[:3], width=200)
        print("\nSome neutral posts:")
        pprint.pprint(list(df[df["label"] == 0]["combined_content"])[:3], width=200)
        print("\nSome negative posts:")
        pprint.pprint(list(df[df["label"] == -1]["combined_content"])[:3], width=200)

        # Frequency distribution of the 20 most common positive & negative words in the text
        frequent_pos_words = FreqDist(df.loc[df["label"] == 1]["combined_content"])
        print(f"\nMost frequent positive words: {frequent_pos_words.values()}")
        frequent_neg_words = FreqDist(df.loc[df["label"] == -1]["combined_content"])
        print(f"Most frequent negative words: {frequent_neg_words.most_common(20)}")


if __name__ == "__main__":
    # nlp_utils.download_nltk_data()
    INPUT_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "posts"
    rb_incidents = ["Assassins-Creed-Unity"]

    for rb_name in rb_incidents:
        incident_folder = INPUT_FOLDER / rb_name
        twitter_data = pd.read_csv(incident_folder / f"twitter_combined_{rb_name}.csv",
                                   nrows=20)  # TODO use all later

        english_data = twitter_data[twitter_data["detected_language"] == "english"]
        # posts_in_rb_time = english_data[english_data["in_rb_time_period"]]   # TODO

        # remove stop words, clean text and lemmatize
        apply_standard_text_preprocessing(english_data, text_col="combined_content", is_social_media_data=True)

        docs_original = list(english_data["combined_content"])
        docs_lemmatized = list(english_data["text_preprocessed"])

        is_reddit_data = False
        source_tag = ""
        if english_data["source"].iloc[0] == "Twitter":
            source_tag = "tw_"
        elif english_data["source"].iloc[0] == "Reddit":
            source_tag = "red_"
            is_reddit_data = True

        # TODO compare different sentiment analysis options
        vader_sentiment_analysis_sentence_level(english_data, "combined_content")
        vader_sentiment_analysis(english_data, "combined_content")
        # vader_sentiment_analysis(english_data, "text_preprocessed")   # compare lemmatized

        textblob_sentiment_analysis(english_data, "combined_content")

        print("debug")
