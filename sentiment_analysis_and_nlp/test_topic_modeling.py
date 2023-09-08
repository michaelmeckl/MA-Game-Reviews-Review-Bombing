#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
import pprint
import re
from collections import defaultdict
import pandas as pd
import spacy
from transformers_test import example_reviews_hogwarts_legacy, example_posts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from bertopic import BERTopic

from useful_code_from_other_projects.utils import enable_max_pandas_display_size


def num2word(d):
    # taken from https://github.com/alfredtangsw/steamvox/blob/master/01_scrape_eda.ipynb
    num_dict = {'0': 'zero',
                '1': 'one',
                '2': 'two',
                '3': 'three',
                # 'i':'one',  Using Roman numeral 'i' will conflict with the pronoun "I", which is not a number
                'ii': 'two',
                'iii': 'three'
                }
    if (len(d) == 1 and d in '0123') or (d in ['ii', 'iii']):
        word = num_dict[d]
    elif len(str(d)) == 1 and str(d) in '0123':
        word = num_dict[str(d)]
    else:
        word = d

    return word


def clean_input_text(text: str) -> list[str]:
    en_stopwords = list(set(STOPWORDS))
    en_stopwords = [w for w in en_stopwords if w not in ['one', 'two', 'three']]  # retain these for making n-grams

    text = text.lower()
    text = re.split(r'\W+', text)

    text = [num2word(w) for w in text]
    text = [word for word in text if
            word not in en_stopwords and 1 < len(word) <= len('pneumonoultramicroscopicsilicovolcanoconiosis')]

    return text


# see https://spacy.io/api/annotation#pos-tagging
def spacy_lemma(input_list, allowed_postags=['NOUN']):
    spacy_nlp = spacy.load("en_core_web_sm")
    lemma_doc_list = [spacy_nlp(" ".join(bow)) for bow in input_list]
    print(lemma_doc_list)
    lemma_text = [token.text if '_' in token.text else token.lemma_ if token.pos_ in allowed_postags else '' for
                  lemma_doc in lemma_doc_list for token in lemma_doc]
    return lemma_text


def test_topic_modeling_lda_sklearn(input_data):
    # Code based on https://www.toptal.com/python/topic-modeling-python
    count_vect = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True)
    x_counts = count_vect.fit_transform(input_data)
    # print(x_counts.todense())

    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(x_counts)

    dimension = 100  # should be fine-tuned
    lda = LDA(n_components=dimension)
    lda_array = lda.fit_transform(x_tfidf)
    # print(lda_array)

    components = [lda.components_[i] for i in range(len(lda.components_))]
    features = count_vect.get_feature_names_out()
    print(features)
    # important_words = [sorted(features, key=lambda x: components[j][features.index(x)], reverse=True)[:3] for j in
    #                    range(len(components))]
    # print(f"\nImportant words: {important_words}")


def test_topic_modeling_gensim(input_data):
    # Code based in parts on https://github.com/alfredtangsw/steamvox/tree/master

    # preprocessing
    cleaned_tokens = [simple_preprocess(text) for text in input_data]
    cleaned_tokens = [num2word(w) for w in cleaned_tokens]
    en_stopwords = list(set(STOPWORDS))
    en_stopwords = [w for w in en_stopwords if w not in ['one', 'two', 'three']]  # retain these for making n-grams
    preprocessed_data = [word for word in cleaned_tokens if word not in en_stopwords]
    print(preprocessed_data)

    # lemmatized_data = spacy_lemma(preprocessed_data)
    # print(lemmatized_data)

    """
    # remove words that appear only once
    frequency = defaultdict(int)
    for text in preprocessed_data:
        for token in text:
            frequency[token] += 1
    texts = [
        [token for token in text if frequency[token] > 1]
        for text in preprocessed_data
    ]
    # pprint.pprint(texts)
    """

    dictionary = corpora.Dictionary(preprocessed_data)
    print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_data]
    # pprint.pprint(corpus)

    use_lsi = True
    if use_lsi:
        # Train Latent Semantic Indexing with 300D vectors.
        num_topics = 300
        # lsi = models.LsiModel(corpus, num_topics=num_topics)
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
        corpus_model = model[corpus_tfidf]
    else:
        num_topics = 100
        model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
        print(model.print_topics(2))
        corpus_model = model[corpus]

    def check_topics(topic_model):
        # Check resulting topics.
        topic_list = topic_model.print_topics(num_topics=num_topics, num_words=15)

        for index, i in enumerate(topic_list):
            str1 = str(i[1])
            for c in "0123456789+*\".":
                str1 = str1.replace(c, "")
            str1 = str1.replace("  ", " ")
            print(str1)

    for doc, as_text in zip(corpus_model, input_data):
        print(doc, as_text)

    check_topics(model)


def test_topic_modeling_bertopic(input_data):
    # TODO remove stop words and maybe do some other preprocessing!
    nlp = spacy.load('en_core_web_sm', exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

    topic_model = BERTopic(embedding_model=nlp, nr_topics="auto")  # calculate_probabilities=True
    topics, probs = topic_model.fit_transform(input_data)
    # for topic, probability in zip(topics, probs):
    #    print(f"{topic} - {probability}")

    print("#############\n")
    # Topic -1 declares outliers that could not be assigned to any topic
    pprint.pprint(topic_model.get_topic_info())
    print("#############\n")
    # show the most frequent topic
    pprint.pprint(topic_model.get_topic(0))
    print("#############\n")
    pprint.pprint(topic_model.get_document_info(input_data))

    # search for topics
    # similar_topics, similarity = topic_model.find_topics("manipulate", top_n=5)
    # pprint.pprint(topic_model.get_topic(similar_topics[0]))

    # TODO this method won't work no matter what I try :( either too few documents or topics probably
    # fig = topic_model.visualize_topics()
    # fig.show()
    fig = topic_model.visualize_barchart()
    # fig.show()


def test_topic_modeling(use_option=2):
    """
    num_rows = 100
    reviews_file = pathlib.Path(__file__).parent.parent / "data_for_analysis" / "steam_user_reviews_Hogwarts_Legacy_old.csv"
    df = pd.read_csv(reviews_file, nrows=num_rows)

    use_more_data = False
    if use_more_data:
        train_data_len = int(num_rows * 0.8)  # 80 - 20 - split
        example_data = list(df['content'][:train_data_len])
        example_data.extend(example_reviews_hogwarts_legacy)
        valid_data = list(df['content'][train_data_len:])
    else:
        example_data = list(df['content'][:10])
        example_data.extend(example_reviews_hogwarts_legacy)
    """

    reviews_file = pathlib.Path(__file__).parent.parent / "data_for_analysis" / "steam" / "steam_user_reviews_cyberpunk_2077.csv"
    df = pd.read_csv(reviews_file)
    example_data = list(df['content'])

    example_data.extend(example_reviews_hogwarts_legacy)
    print(str(len(example_data)) + " examples")
    # pprint.pprint(example_data)

    match use_option:
        case 0:
            print("Using Sklearn LDA ...")
            test_topic_modeling_lda_sklearn(example_data)
        case 1:
            print("Using Gensim ...")
            test_topic_modeling_gensim(example_data)
        case 2:
            print("Using BERTopic ...")
            test_topic_modeling_bertopic(example_data)
        case _:
            print("No suitable option found!")


if __name__ == "__main__":
    enable_max_pandas_display_size()
    test_topic_modeling()
