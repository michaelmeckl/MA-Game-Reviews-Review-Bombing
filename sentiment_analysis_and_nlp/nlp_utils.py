import re
import string
import collections
import itertools
from nltk import word_tokenize, RegexpTokenizer, bigrams, FreqDist
from nltk.corpus import stopwords
from rake_nltk import Rake
import pandas as pd
from textblob import TextBlob


_SOME_ALPHA_RE = re.compile(r'[A-Za-z]+')
_ONLY_ALPHA_RE = re.compile(r'^[A-Za-z]*$')

punctuation_list = list(string.punctuation) + ['`', '’', '…']


def filter_paragraph(p):
    """ Function taken from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/wikisum/utils.py#L214

    Simple filter to remove obviously bad paragraphs (bad text extraction).
    Note this needs to run very quickly as it is applied to every paragraph
    in the corpus, so nothing fancy! This whole method should be linear
    expected time in len(p).

    Args:
      p: string, paragraph

    Returns:
      True if we should remove the paragraph.
    """
    # Expect a minimum number of words.
    tokens = p.split()
    if len(tokens) < 6:
        return True

    # Require some letters.
    if not re.search(_SOME_ALPHA_RE, p):
        return True

    # Keep this one at the end, probably the most complicated logic.
    # We try to detect sentences, which should have a minimum of 3 tokens
    # with only alphabetic characters.
    last = 0
    found_sentence = False
    num_alpha = 0
    for i, x in enumerate(tokens):
        if x == '.':
            if i - last > 3 and num_alpha >= 3:
                found_sentence = True
                break
            last = i
            num_alpha = 0
        if re.match(_ONLY_ALPHA_RE, x):
            num_alpha += 1
    if not found_sentence:
        return True

    return False


# try to identify "low-effort" comments and posts
def identify_low_effort(text):
    without_punct = [char for char in text if char not in string.punctuation]  # remove punctuation
    without_punct = "".join(without_punct)
    tokens = word_tokenize(without_punct)  # split into list of words
    if len(tokens) < 6 or len(without_punct) < 140:
        label = "low effort"
    else:
        label = "high effort"
    return label


def normalize_text(text):
    text = text.lower()
    # remove html markup
    text = re.sub("(<.*?>)", "", text)
    # remove non-ascii and digits
    text = re.sub("(\\W|\\d)", " ", text)
    # Space around punctuation
    text = re.sub("[%s]" % re.escape(string.punctuation), r" \g<0> ", text)
    text = re.sub(r"\s+", " ", text)
    # remove whitespace
    text = text.strip()
    return text


def preprocess_text(text):
    # method taken from https://github.com/MullerAC/video-game-review-analysis/
    text = re.sub(r'\[.*?\]', '', text)  # remove markdown tags, only needed for Steam reviews
    text = text.translate(str.maketrans('', '', ''.join(punctuation_list)))  # remove all punctuation
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  # tokenize words with only numbers and latin characters
    return tokenizer.tokenize(text.lower())


def apply_text_preprocessing(df: pd.DataFrame, text_col: str):
    # apply some basic text processing on a specific column of a dataframe (e.g. review_text)
    X = df[text_col].to_numpy()
    processed_column = list(map(preprocess_text, X))
    return processed_column


def remove_url(txt):
    # Taken from
    # https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/calculate-tweet-word-frequencies-in-python/
    """Replace URLs found in a text string with nothing (i.e. it will remove the URL from the string).
    """
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


def get_most_frequent_words(text_list: list[str]):
    # see https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/calculate-tweet-word
    # -bigrams/
    texts_no_urls = [remove_url(text) for text in text_list]
    words_in_text = [text.lower().split() for text in texts_no_urls]

    stop_words = set(stopwords.words('english'))
    text_without_stop = [[word for word in text_words if word not in stop_words]
                         for text_words in words_in_text]

    # List of all words across all texts
    all_words = list(itertools.chain(*text_without_stop))
    # Create counter
    counts = collections.Counter(all_words)
    print(counts.most_common(15))

    terms_bigram = [list(bigrams(text)) for text in text_without_stop]
    # Flatten list of bigrams in clean texts
    bigram_list = list(itertools.chain(*terms_bigram))
    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigram_list)
    print(bigram_counts.most_common(20))


def check_contains_most_frequent_words(document_words: list, all_documents):
    all_words = FreqDist(word.lower() for word in all_documents)
    word_features = list(all_words)[:500]  # 500 most frequent words

    document_words = set(document_words)   # we convert it to a set because it is much faster than a list
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features


def get_pos_tags(text: str):
    blob = TextBlob(text)
    pos_tags = blob.tags
    return pos_tags


def textblob_utils(text: str):
    blob = TextBlob(text)
    print(blob.words)  # split into words
    print(blob.sentences)  # split into sentences
    lemmata = [word.lemmatize() for word in blob.words]
    print(lemmata)


def test_semantic_similarity():
    """
    See http://swoogle.umbc.edu/SimService/api.html for more information.
    """
    from requests import get

    sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"
    textual_similarity_service = "http://swoogle.umbc.edu/StsService/GetStsSim"

    def sss(s1, s2, type='relation', corpus='webbase'):
        try:
            response = get(sss_url,
                           params={'operation': 'api', 'phrase1': s1, 'phrase2': s2, 'type': type, 'corpus': corpus})
            return float(response.text.strip())
        except Exception as e:
            print(f'Error in getting similarity for {(s1, s2)}: {e}')
            return 0.0

    print(f"Testing semantic similarity: {sss('review bomb', 'controversial')}")


def extract_keywords(text: str):
    """
    Uses the RAKE (Rapid Automatic Keyword Extraction) algorithm to extract the most suitable keywords from a text.
    See https://csurfer.github.io/rake-nltk/_build/html/index.html
    """
    # Uses stopwords for english from NLTK, and all punctuation characters by default
    r = Rake()
    r.extract_keywords_from_text(text)
    for score, phrase in r.get_ranked_phrases_with_scores():
        if score > 5:
            print(f"Score: {score}, Phrase:{phrase}\n")


def extract_keywords_with_tf_idf(df: pd.DataFrame):
    # Extract the most common keywords from a large number of texts by calculating the tf-idf.
    # See https://github.com/kavgan/nlp-in-practice/blob/master/tf-idf
    pass  # TODO


def clean_tweet(tweet):
    # Taken from https://www.kaggle.com/code/j13mehul/bert-fine-tuning-using-peft#Freezing-the-head
    # Use like: df['text'] = df['text'].apply(lambda s: clean(s))

    # Special characters
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)

    Special = '@#!?+&*[]-%:/()$=><|{}^'
    for s in Special:
        tweet = tweet.replace(s, "")

    return tweet
