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
    Cutoffs were chosen according to the official VADER github repository: https://github.com/cjhutto/vaderSentiment/tree/master
    """
    if score > 0.05:
        class_result = "positive", 1
    elif score < -0.05:
        class_result = "negative", -1
    else:
        class_result = "neutral", 0
    # return class_result[1] if result_as_number else class_result[0]
    return class_result


def textblob_sentiment_analysis(df, df_text_col):
    def get_textblob_sentiment(post):
        blob = TextBlob(post)
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

    df['sentiment_score_whole'] = df[df_text_col].apply(lambda x: get_sentiment_score(x))
    # create a new column to classify the polarity scores
    df[["label_text_whole", "label_whole"]] = df["sentiment_score_whole"].apply(lambda score: pd.Series(classify_sentiment_score(
        score)))

    sentiment_mean = df['sentiment_score_whole'].mean()
    print(f"Mean sentiment score for all texts: {sentiment_mean:.2f}")
    print(df["label_whole"].value_counts())
    print(df["label_whole"].value_counts(normalize=True) * 100)  # show label distribution (i.e. as percentage)

    # Plot a histogram of sentiment scores
    plt.hist(df['sentiment_score_whole'], bins=20, color='blue', alpha=0.5)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Vader Sentiment Scores')
    # plt.show()


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


def compare_sentiment_analysis_results():
    INPUT_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "posts"

    for rb_name in rb_incidents:
        incident_folder = INPUT_FOLDER / rb_name
        twitter_data = pd.read_csv(incident_folder / f"twitter_combined_{rb_name}.csv",
                                   nrows=30)
        reddit_submission_data = pd.read_csv(incident_folder / f"reddit_submissions_combined_{rb_name}.csv",
                                             nrows=30)
        reddit_comment_data = pd.read_csv(incident_folder / f"combined_reddit_comments_{rb_name}.csv",
                                          nrows=30)

        english_twitter_data = twitter_data[twitter_data["detected_language"] == "english"]
        english_reddit_submissions = reddit_submission_data[reddit_submission_data["detected_language"] == "english"]
        english_reddit_comments = reddit_comment_data[reddit_comment_data["detected_language"] == "english"]

        # remove stop words, clean text and lemmatize
        apply_standard_text_preprocessing(english_twitter_data, text_col="combined_content", remove_punctuation=False,
                                          remove_stopwords=False, is_social_media_data=True)
        apply_standard_text_preprocessing(english_reddit_submissions, text_col="combined_content",
                                          remove_punctuation=False,
                                          remove_stopwords=False, is_social_media_data=True)
        apply_standard_text_preprocessing(english_reddit_comments, text_col="combined_content",
                                          remove_punctuation=False,
                                          remove_stopwords=False, is_social_media_data=True)

        use_cleaned = False
        if use_cleaned:
            col = "text_cleaned"
        else:
            # use original text
            col = "combined_content"

        for data in [english_twitter_data, english_reddit_submissions, english_reddit_comments]:
            # compare different sentiment analysis options
            vader_sentiment_analysis_sentence_level(data, col)
            vader_sentiment_analysis(data, col)
            # textblob_sentiment_analysis(data, col)

            # create small df with only text columns and sentiment results for comparison
            subset_df = data[[col, "sentiment_score_sentence_level", "sentiment_score_whole"]]
            print("debug")

        # test on review data for this incident too
        # choose 30 reviews randomly for testing the sentiment analysis
        sampled_review_data = review_data[review_data["review_bombing_incident"] == rb_name].sample(frac=1, random_state=42).head(30)
        apply_standard_text_preprocessing(sampled_review_data, text_col="review",
                                          remove_punctuation=True, remove_stopwords=False)
        col = "text_cleaned" if use_cleaned else "review"
        vader_sentiment_analysis_sentence_level(sampled_review_data, col)
        vader_sentiment_analysis(sampled_review_data, col)
        # textblob_sentiment_analysis(sampled_review_data, col)
        subset_review_df = sampled_review_data[[col, "sentiment_score_sentence_level", "sentiment_score_whole"]]

        print("debug outer")


def apply_sentiment_analysis(df: pd.DataFrame, text_col: str, col_for_sentiment_analysis: str,
                             perform_preprocessing=False, social_media_data=True):
    # based on tests, preprocessing usually does not improve much (or makes it even worse)
    if perform_preprocessing:
        apply_standard_text_preprocessing(df, text_col=text_col, remove_stopwords=False, remove_punctuation=False,
                                          is_social_media_data=social_media_data)

    vader_sentiment_analysis_sentence_level(df, col_for_sentiment_analysis)
    # vader_sentiment_analysis(df, col_for_sentiment_analysis)
    df.drop(columns=['label_text'], axis=1, inplace=True)
    # return df


if __name__ == "__main__":
    # rb_incidents = ["Skyrim-Paid-Mods", "Assassins-Creed-Unity", "Firewatch", "Mortal-Kombat-11",
    #                 "Borderlands-Epic-Exclusivity", "Ukraine-Russia-Conflict"]
    rb_incidents = ["Borderlands-Epic-Exclusivity", "Assassins-Creed-Unity"]  # for faster testing

    review_data = pd.read_csv(pathlib.Path(__file__).parent.parent /
                              "combined_final_annotation_all_projects_updated.csv")

    compare_sentiment_analysis_results()
