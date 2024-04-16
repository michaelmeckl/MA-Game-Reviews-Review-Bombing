import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


def download_nltk_data():
    # only do this once at the beginning if these aren't installed yet!
    nltk.download([
        "stopwords",
        "wordnet",
        # "averaged_perceptron_tagger",
        "vader_lexicon",  # vader is a sentiment lexicon specifically for social media data
        "punkt",
    ])


def stem_text(words: list[str]):
    porter_stemmer = PorterStemmer()  # SnowballStemmer()
    stemmed_words = [porter_stemmer.stem(word=word) for word in words]
    return stemmed_words


def lemmatize_text(text: str | list[str]):
    """
    Stemming might return a root word that is not an actual word whereas lemmatizing returns a root word that is an
    actual word.
    """
    lemmatizer = WordNetLemmatizer()
    if isinstance(text, str):
        text = text.lower()
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    else:
        lemmatized_words = [lemmatizer.lemmatize(w.lower()) for w in text]

    return lemmatized_words, ' '.join(lemmatized_words)


def get_nltk_stop_words():
    return nltk.corpus.stopwords.words("english")


def remove_stopwords(text: str | list[str]):
    if isinstance(text, str):
        text_split = text.split()
    else:
        text_split = text
    stopword_list = set(nltk.corpus.stopwords.words("english"))
    words = [word for word in text_split if word.lower() not in stopword_list]
    return " ".join(words)


def split_into_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text, 'english')
    new_sentences = [split_sent for sent in sentences for split_sent in sent.split("\n")]
    return new_sentences


def tokenize_text(text, is_social_media_data=False):
    if is_social_media_data:
        # uses NLTK's own tweet tokenizer to tokenize tweets
        tokenizer = TweetTokenizer(reduce_len=True)
        return tokenizer.tokenize(text)
    else:
        # there are probably better ways to do this; RegexpTokenizer also ignores punctuation compared to this for example
        # see https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/
        return nltk.word_tokenize(text)


"""
def remove_unwanted_words(text):
    word_list = tokenize_text(text)
    # remove stopwords
    new_word_list = remove_stopwords(word_list)
    # include only the words that are made up of letters with str.isalpha()
    new_word_list = [word for word in new_word_list if word.isalpha()]
    return new_word_list
"""
