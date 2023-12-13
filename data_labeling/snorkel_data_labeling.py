#!/usr/bin/python
# -*- coding: utf-8 -*-

import pprint
import pathlib
import re
import random
import pandas as pd
from typing import List
from snorkel.preprocess import preprocessor
from snorkel.labeling import LFAnalysis
from textblob import TextBlob
from analyze_reviews import combine_metacritic_steam_reviews
from sentiment_analysis_and_nlp.nlp_utils import detect_language
from useful_code_from_other_projects import utils
from useful_code_from_other_projects.utils import enable_max_pandas_display_size
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from snorkel.labeling import LabelingFunction, labeling_function
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier


# label mappings
ABSTAIN = -1
NOT_REVIEW_BOMB = 0
REVIEW_BOMB = 1

DATA_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis"


def load_unlabeled_data() -> pd.DataFrame:
    steam_review_data = pd.read_csv(DATA_FOLDER / "steam" / "steam_user_reviews_Borderlands_2.csv")
    steam_general_game_info = pd.read_csv(DATA_FOLDER / "steam" / "steam_general_info_Borderlands_2.csv")
    metacritic_review_data = pd.read_csv(DATA_FOLDER / "metacritic" / "user_reviews_pc_borderlands-2.csv")
    metacritic_game_info = pd.read_csv(DATA_FOLDER / "metacritic" / "game_info_pc_borderlands-2.csv")

    # take only very negative metacritic ratings
    negative_metacritic_review_data = metacritic_review_data[metacritic_review_data["rating"] < 3]
    num_samples = 50 if len(negative_metacritic_review_data) > 50 else len(negative_metacritic_review_data)
    filtered_metacritic = negative_metacritic_review_data.sample(n=num_samples, random_state=1)

    filtered_steam = steam_review_data.sample(n=50, random_state=1)
    filtered_steam = utils.remove_linebreaks_from_pd_cells(filtered_steam, column_name="content")

    combined_review_df = combine_metacritic_steam_reviews(filtered_steam, filtered_metacritic,
                                                          steam_general_game_info, metacritic_game_info)

    # remove non-english reviews for now; unfortunately, this removes some very short english reviews or the ones
    # with many links as well
    #english_review_data = combined_review_df[combined_review_df["review"].apply(lambda x: detect_language(x)).eq('en')]

    # TODO split ?
    # train_data_len = int(len(english_review_data) * 0.8)  # 80 - 20 - split
    # train_data = list(english_review_data['review'][:train_data_len])

    # random shuffle data
    shuffled_data = combined_review_df.sample(frac=1)
    return shuffled_data


def make_keyword_lf(keywords: List[str], label: str, field: str = "text"):
    """
    Method taken from https://github.com/snorkel-team/snorkel-zoo
    Can be used like this:
        keyword_please = make_keyword_lf(keywords=["please", "plz"], label="SPAM")
    """
    def keyword_lookup(x, keywords, label):
        if any(word in x[field].lower() for word in keywords):
            return label
        return ABSTAIN

    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.review)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x


@labeling_function(pre=[textblob_sentiment])
def lf_textblob_polarity(x):
    return NOT_REVIEW_BOMB if x.polarity > 0.9 else ABSTAIN


@labeling_function()
def lf_keywords(x):
    """Many borderlands RB reviews talk about anything related to the Epic Games Store exclusivity."""
    keywords = ["epic store", "epic games", "exclusive", "exclusivity"]
    return REVIEW_BOMB if any(word in x.review.lower() for word in keywords) else ABSTAIN


@labeling_function()
def lf_regex(x):
    return REVIEW_BOMB if re.search(r"epic\s*(store|games|exclusiv)", x.review, flags=re.I) else ABSTAIN


@labeling_function()
def lf_contains_review_bomb(x):
    """If the Review Bomb is explicitly mentioned."""
    return REVIEW_BOMB if re.search(r"review\s*bomb", x.review, flags=re.I) else ABSTAIN


@labeling_function()
def lf_metadata(x):
    """Likely part of a RB if Steam playtime very short or Metacritic rating 0 or 1."""
    # TODO add other metadata conditions as well such as author information or profile not old enough; review date
    #  during review bombing incident; etc.
    short_playtime = x["author_playtime_at_review_min"] < 60
    low_metacritic_rating = x["rating"] < 2
    return REVIEW_BOMB if low_metacritic_rating or short_playtime else ABSTAIN


def train_snorkel_label_model(df_train, df_test):
    # TODO
    df_train["label"] = 0

    lfs = [lf_keywords, lf_contains_review_bomb, lf_textblob_polarity]
    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)
    L_test = applier.apply(df_test)

    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)   # cardinality 2 because binary classification
    label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")
    pprint.pprint(df_train)

    Y_test = df_test.label.values
    label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
    print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

    # TODO currently only labeled either as Review Bomb (1) or ABSTAIN (-1); LFs for 0 are needed !
    #  -> lb_textblob_polarity is not enough

    df_train_filtered = df_train[df_train.label != ABSTAIN]  # remove unlabeled entries
    return df_train_filtered


def test_snorkel_data_labeling(unlabeled, labeled):
    """
    lfs = [lf_keywords, lf_contains_review_bomb, lf_metadata]
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(unlabeled)

    # show some statistics for the LFs
    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())
    # check example labels for the first and last Labeling Function
    example_rb_labels_0 = unlabeled.iloc[L_train[:, 0] == REVIEW_BOMB].sample(10, random_state=1)
    pprint.pprint(example_rb_labels_0)
    example_rb_labels_2 = unlabeled.iloc[L_train[:, 2] == REVIEW_BOMB].sample(10, random_state=1)
    pprint.pprint(example_rb_labels_2)
    """

    new_train_data = train_snorkel_label_model(unlabeled, labeled)

    """
    # train a discriminative classifier on the newly labeled data
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(new_train_data.review.tolist())
    X_test = vectorizer.transform(labeled.review.tolist())

    clf = LogisticRegression(solver="lbfgs")   # C=1e3, solver="liblinear"
    clf.fit(X=X_train, y=new_train_data.label.values)
    print(f"Test Accuracy: {clf.score(X=X_test, y=labeled.label.values) * 100:.1f}%")
    """


if __name__ == "__main__":
    enable_max_pandas_display_size()
    random.seed(42)

    labeled_data = pd.read_csv("labeled_data.csv")
    """
    data = load_unlabeled_data()
    unlabeled_data_unique = data[~data["review"].isin(labeled_data["review"])]
    unlabeled_data_unique.to_csv("unlabeled_data.csv", index=False)
    """

    unlabeled_data = pd.read_csv("unlabeled_data.csv")
    test_snorkel_data_labeling(unlabeled_data, labeled_data)
