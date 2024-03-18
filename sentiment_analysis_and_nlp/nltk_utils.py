import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sentiment_analysis_and_nlp.nlp_utils import punctuation_list


def download_nltk_data():
    # only do this once at the beginning if these aren't installed yet!
    nltk.download([
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "vader_lexicon",  # vader is a sentiment lexicon specifically for social media data
        "punkt",
    ])


def stem_text(words: list[str]):
    porter_stemmer = PorterStemmer()
    stemmed_words = [porter_stemmer.stem(word=word) for word in words]
    return stemmed_words


def lemmatize_text(text: str | list[str]) -> list[str]:
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

    # return ' '.join(lemmatized_words)
    return lemmatized_words


def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def remove_stopwords(text):
    stopword_list = nltk.corpus.stopwords.words("english") + punctuation_list
    words = [word for word in text if word.lower() not in stopword_list]
    return words


def split_into_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text, 'english')
    new_sentences = [split_sent for sent in sentences for split_sent in sent.split("\n")]
    return new_sentences


def split_into_words(text):
    # there are probably better ways to do this; RegexpTokenizer also ignores punctuation compared to this for example
    # see https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/
    return nltk.word_tokenize(text)


def split_sentence_into_words(sentence, normalize=True):
    # find all tokens and optionally stem them
    tokens = re.findall(r'[^\s!,.?":;0-9]+', sentence.lower())
    if normalize:
        stemmer = SnowballStemmer('english')
        normalized_tokens = [stemmer.stem(token) for token in tokens]
        return normalized_tokens
    else:
        return tokens


def remove_unwanted_words(text):
    word_list = split_into_words(text)
    # remove stopwords
    new_word_list = remove_stopwords(word_list)
    # include only the words that are made up of letters with str.isalpha()
    new_word_list = [word for word in new_word_list if word.isalpha()]
    return new_word_list


def tokenize_tweets(text):
    # uses NLTK's own tweet tokenizer to tokenize tweets
    tokenizer = TweetTokenizer(reduce_len=True)
    return tokenizer.tokenize(text)
