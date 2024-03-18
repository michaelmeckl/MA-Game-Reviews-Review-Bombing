#!/usr/bin/python
# -*- coding: utf-8 -*-
import pprint
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize, FreqDist


# TODO test other sentiment analysis tools like TextBlob or even embedding-models like FLAIR too!
#  -> TextBlob probably worse, FLAIR should perform a lot better (but takes also a lot more time)
def test_reddit_sentiment_analysis(debug=True):
    sia = SentimentIntensityAnalyzer()
    results = []

    # load data from csv
    df = pd.read_csv("../data_for_analysis/reddit/merged_reddit_submissions_Cyberpunk2077.csv")
    title_data = list(df['title'])
    content_data = list(df['content'])

    # combine title and content text
    data = title_data
    # split content into sentences first so the polarity_scores function works
    lines_list = tokenize.sent_tokenize(content_data[0])
    data.extend(lines_list)

    for sentence in data:
        polarity_score = sia.polarity_scores(sentence)   # only works on word or sentence level (?)
        if debug:
            print(f"\nSentence: \n{sentence}")
            for field in sorted(polarity_score):
                print('{0}: {1}, '.format(field, polarity_score[field]), end='')

        polarity_score["text"] = sentence
        results.append(polarity_score)

    print("\n #################### \n")

    # use from_records() as we have a list of dictionaries
    sentiment_df = pd.DataFrame.from_records(results)

    # create a new column to classify the polarity scores
    sentiment_df["label"] = 0
    sentiment_df.loc[sentiment_df["compound"] > 0.2, "label"] = 1
    sentiment_df.loc[sentiment_df["compound"] < -0.2, "label"] = -1
    sentiment_df.loc[(sentiment_df["compound"] > -0.2) & (sentiment_df["compound"] < 0.2), "label"] = 0

    if debug:
        print(sentiment_df.head(5))

    # show some sentiment analysis results
    if debug:
        print(sentiment_df.label.value_counts())
        print(sentiment_df.label.value_counts(normalize=True) * 100)  # show label distribution (i.e. as percentage)

        print("\nSome positive posts:")
        pprint.pprint(list(sentiment_df[sentiment_df["label"] == 1].text)[:3], width=200)
        print("\nSome neutral posts:")
        pprint.pprint(list(sentiment_df[sentiment_df["label"] == 0].text)[:3], width=200)
        print("\nSome negative posts:")
        pprint.pprint(list(sentiment_df[sentiment_df["label"] == -1].text)[:3], width=200)

        # Frequency distribution of the 20 most common positive & negative words in the text
        frequent_pos_words = FreqDist(sentiment_df.loc[sentiment_df["label"] == 1].text)
        print(f"\nMost frequent positive words: {frequent_pos_words}")
        print(sum(frequent_pos_words.values()))

        frequent_neg_words = FreqDist(sentiment_df.loc[sentiment_df["label"] == -1].text)
        print(f"Most frequent negative words: {frequent_neg_words.most_common(20)}")

    new_df = sentiment_df[["label", "text"]]
    new_df.to_csv("reddit_test_sentiment_results.csv", index=False)


if __name__ == "__main__":
    # nlp_utils.download_nltk_data()
    test_reddit_sentiment_analysis()
