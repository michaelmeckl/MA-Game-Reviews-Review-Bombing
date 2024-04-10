import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import multiprocessing


stopwords_list = stopwords.words("english")


def remove_stopwords(text):
    text_tokens = text.split(" ")
    final_list = [word for word in text_tokens if word not in stopwords_list]
    text = ' '.join(final_list)
    return text


def clean_data(text):
    text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)  # replace everything except the provided characters
    text = re.sub(r'[\\/×^\]\[÷]', '', text)
    return text


def get_w2vdf(df, column):
    w2v_df: list = pd.DataFrame(df[column]).values.tolist()
    for i in range(len(w2v_df)):
        w2v_df[i] = w2v_df[i][0].split(" ")
    return w2v_df


def train_w2v(w2v_df):
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=4,
                         window=4,
                         vector_size=300,
                         alpha=0.03,
                         min_alpha=0.0007,
                         sg=1,  # 1 means use skip-gram, otherwise uses cbow
                         workers=cores - 1)

    w2v_model.build_vocab(w2v_df, progress_per=10000)
    w2v_model.train(w2v_df, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)
    return w2v_model


# Code taken from https://towardsdatascience.com/word-embedding-techniques-word2vec-and-tf-idf-explained-c5d02e34d08
def apply_word2vec(df: pd.DataFrame, column_name: str):
    df[[column_name]] = df[[column_name]].astype(str)
    df[column_name] = df[column_name].map(lambda text: text.lower())
    df[column_name] = df[column_name].apply(clean_data)
    df[column_name] = df[column_name].apply(remove_stopwords)

    w2v_df = get_w2vdf(df, column_name)
    w2v_model = train_w2v(w2v_df)
    return w2v_model
