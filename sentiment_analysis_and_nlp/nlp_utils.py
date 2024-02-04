import re
import string
from string import punctuation
import collections
import itertools
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from rake_nltk import Rake
import spacy
import spacy_fastlang  # noqa: F401 # pylint: disable=unused-import
import pandas as pd
from langdetect import detect, LangDetectException, DetectorFactory, detect_langs
from textblob import TextBlob

# make langdetect deterministic, see https://pypi.org/project/langdetect/
DetectorFactory.seed = 0

_SOME_ALPHA_RE = re.compile(r'[A-Za-z]+')
_ONLY_ALPHA_RE = re.compile(r'^[A-Za-z]*$')

punctuation_list = list(punctuation) + ['`', '’', '…']


def download_nltk_data():
    # only do this once at the beginning if these aren't installed yet!
    nltk.download([
        "stopwords",
        "wordnet",
        # "twitter_samples",
        # "movie_reviews",
        "averaged_perceptron_tagger",
        "vader_lexicon",  # vader is a sentiment lexicon specifically for social media data
        "punkt",
    ])


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


def stem_text(words: list[str]):
    porter_stemmer = PorterStemmer()
    stemmed_words = [porter_stemmer.stem(word=word) for word in words]
    return stemmed_words


def lemmatize_text(text):
    """
    Stemming might return a root word that is not an actual word whereas lemmatizing returns a root word that is an
    actual word.
    """
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    text = ' '.join(words)
    return text


def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words("english") + punctuation_list
    words = [word for word in text if word.lower() not in stopwords]
    return words


def split_into_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text, 'english')
    new_sentences = [split_sent for sent in sentences for split_sent in sent.split("\n")]
    return new_sentences


def split_sentence_into_words(sentence, normalize=True):
    # find all tokens and optionally stem them
    tokens = re.findall(r'[^\s!,.?":;0-9]+', sentence.lower())
    if normalize:
        stemmer = SnowballStemmer('english')
        normalized_tokens = [stemmer.stem(token) for token in tokens]
        return normalized_tokens
    else:
        return tokens


def split_text_into_words(text):
    # there are probably better ways to do this; RegexpTokenizer also ignores punctuation compared to this for example
    # see https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/
    return nltk.word_tokenize(text)


def remove_unwanted_words(text):
    word_list = split_text_into_words(text)
    # remove stopwords
    new_word_list = remove_stopwords(word_list)
    # include only the words that are made up of letters with str.isalpha()
    new_word_list = [word for word in new_word_list if word.isalpha()]
    return new_word_list


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


def tokenize_tweets(text):
    # uses NLTK's own tweet tokenizer to tokenize tweets
    tokenizer = TweetTokenizer(reduce_len=True)
    return tokenizer.tokenize(text)


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


def check_contains_most_frequent_words(document_words: list, all_documents):
    all_words = nltk.FreqDist(word.lower() for word in all_documents)
    word_features = list(all_words)[:500]  # 500 most frequent words

    document_words = set(document_words)   # we convert it to a set because it is much faster than a list
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features


def is_token_allowed(token):
    if not token or not token.string.strip() or token.is_stop or token.ispunct:
        return False
    return True


def named_entity_recognition_with_spacy(input_text: str, spacy_nlp):
    nlp_text = spacy_nlp(input_text)
    # lemmatize the text with spacy first
    complete_filtered_tokens = [
        token.lemma_.strip().lower() for token in nlp_text if is_token_allowed(token)
    ]
    lemmatized_sentence = " ".join(complete_filtered_tokens)
    print(f"Lemmatized Sentence: {lemmatized_sentence}")

    # named entity recognition, TODO find out which others exist and which are useful for me
    nlp_text = spacy_nlp(lemmatized_sentence)
    for entity in nlp_text.ents:
        if entity.label_ == "ORG":
            print("(Organization)", entity.text)
        elif entity.label_ == "PERSON":
            print("(Person)", entity.text)
        elif entity.label_ == "GPE":
            print("(Geographical location)", entity.text)
        elif entity.label_ == "EVENT":
            print("(Event)", entity.text)
        elif entity.label_ == "PRODUCT":
            print("(Product)", entity.text)
        elif entity.label_ == "NORP":
            print("(Nationalities, religious, political groups)", entity.text)
        else:
            print("Other entity", entity.text)


def setup_spacy():
    # spacy.download("en_core_web_sm")
    spacy_nlp = spacy.load("en_core_web_sm")
    spacy_stopwords = spacy_nlp.Defaults.stop_words
    # print("All default stopwords in spacy: ", spacy_stopwords)


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


def detect_language(x):
    try:
        return detect(x)
    except (LangDetectException, TypeError):
        return 'unknown'


def detect_contains_english(x):
    try:
        languages = detect_langs(x)
        print(f"\nDetected languages \"{languages}\" for text: \"{x}\"")
        return True if 'en' in languages else False
    except (LangDetectException, TypeError):
        return False


def setup_spacy_language_detection():
    # setup spacy language detector
    spacy_en = spacy.load("en_core_web_sm")
    spacy_en.add_pipe("language_detector")
    return spacy_en


def detect_language_spacy(x, spacy_nlp):
    # a bit less restrictive than lang_detect above but where lang_detect removes too much, spacy_fastlang includes
    # some obvious non-english texts as well ...
    try:
        doc = spacy_nlp(x)
        # noinspection PyProtectedMember
        detected_language = doc._.language
        return detected_language
    except Exception as e:
        print(f"Error while trying to detect language with Spacy: {e}")
        return 'unknown'
