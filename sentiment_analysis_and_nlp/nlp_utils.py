import re
import string
import collections
import itertools
from nltk import word_tokenize, RegexpTokenizer, bigrams, FreqDist
from nltk.corpus import stopwords
from rake_nltk import Rake
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from textblob import TextBlob
from gensim.models import Phrases
from sentiment_analysis_and_nlp import nltk_utils
from sentiment_analysis_and_nlp.spacy_utils import SpacyUtils
from num2words import num2words


punctuation_list = list(string.punctuation) + ['`', '’', '…']

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def apply_standard_text_preprocessing(df: pd.DataFrame, text_col: str, remove_stopwords=True, use_spacy=True,
                                      is_social_media_data=False):
    spacy_utils = SpacyUtils()
    df[text_col] = df[text_col].astype(str)
    df['text_cleaned'] = df[text_col].apply(lambda text: clean_text(text, spacy_utils, remove_stopwords))

    if use_spacy:
        df['text_preprocessed'] = df['text_cleaned'].apply(lambda text: apply_lemmatization_spacy(text, spacy_utils))
    else:
        # use nltk instead of spacy
        df['text_preprocessed'] = df['text_cleaned'].apply(lambda text: apply_lemmatization_nltk(text, is_social_media_data))
    return df


def apply_lemmatization_spacy(text: str, spacy_utils):
    # tokenized_text = spacy_utils.split_into_words(text)
    # combined_tokenized_text = " ".join([token.text for token in tokenized_text])
    lemmatized_token_list, lemmatized_text = spacy_utils.lemmatize_text(text)
    return lemmatized_text


def apply_lemmatization_nltk(text: str, is_social_media_data):
    tokenized_text = nltk_utils.tokenize_text(text, is_social_media_data)
    lemmatized_token_list, lemmatized_text = nltk_utils.lemmatize_text(tokenized_text)
    return lemmatized_text


def clean_text(text: str, spacy_utils: SpacyUtils, remove_stopwords: bool, remove_punctuation=False):
    """
    Adapted from https://github.com/Idilismiguzel/NLP-with-Python/blob/master/Text-Classification.ipynb
    Use like this:
    ```
    df['text'] = df['text'].astype(str)
    df['text_cleaned'] = list(map(clean_text, df['text']))
    ```
    Remove unwanted characters, and format the text.
    """
    # Convert words to lower case
    text = text.lower().strip()

    # Replace contractions with their longer forms
    text_cleaned = text.split()
    new_text = []
    for word in text_cleaned:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)

    # remove stop words
    if remove_stopwords:
        stop_words = combine_stopword_lists(spacy_utils)
        text_cleaned = " ".join([w for w in new_text if w not in stop_words])
    else:
        text_cleaned = " ".join(new_text)

    # Format words and remove unwanted characters
    # text_cleaned = re.sub(r'https?:\/\/.*[\r\n]*', '', text_cleaned, flags=re.MULTILINE)   # remove url
    text_cleaned = re.sub(r'(?:www|https?)\S+', '', text_cleaned, flags=re.MULTILINE)   # better remove url
    text_cleaned = re.sub(r'\<a href', ' ', text_cleaned)
    text_cleaned = re.sub(r'&amp;', '', text_cleaned)
    # text_cleaned = re.sub(r'["\-;%(){}^|+&=*.,!?:#$@\[\]/]', ' ', text_cleaned)
    # replace duplicate special characters:
    text_cleaned = re.sub(r'([!()\-{};:,<>./?@#$%\^&*_~]){2,}', lambda x: x.group()[0], text_cleaned)
    text_cleaned = re.sub(r'<br />', ' ', text_cleaned)
    text_cleaned = re.sub(r'\'', ' ', text_cleaned)
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned)

    # replace numbers with words (i.e. 1 with "one")
    text_cleaned = re.sub(r"(\d+)", lambda number: num2words(int(number.group(0))), text_cleaned)

    if remove_punctuation:
        # remove string.punctuation characters
        text_cleaned = text_cleaned.translate(str.maketrans('', '', ''.join(punctuation_list)))

    # remove single characters surrounded by whitespaces after the cleaning ? such as " i t " -> ""
    # text_cleaned = re.sub(r'\s+.\s+', '', text_cleaned)
    return text_cleaned.strip()


def combine_stopword_lists(spacy_utils: SpacyUtils):
    # combine stopwords from nltk, spacy and gensim into one set
    from gensim.parsing.preprocessing import STOPWORDS
    stop_words = spacy_utils.get_stopwords().union(STOPWORDS).union(nltk_utils.get_nltk_stop_words())
    # retain some words for making n-grams
    stopwords_set = set(stop_words) - {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
    return stopwords_set


def create_bigrams(df, text_col):
    # Taken from https://github.com/Idilismiguzel/NLP-with-Python/blob/master/Text-Classification.ipynb
    # apply this after the method above
    # trigrams with ngram_range=[3,3] or bigrams and trigrams with [2, 3]
    bigram_converter = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[2, 2], lowercase=False)
    bigram_bow = bigram_converter.fit_transform(df[text_col])
    # print(bigram_converter.get_feature_names_out())

    # convert to tf-idf
    tfidf_transform = TfidfTransformer(norm=None)
    bigram_tfidf = tfidf_transform.fit_transform(bigram_bow)
    return bigram_tfidf


def create_bigrams_gensim(texts: list[str]):
    # Taken from https://dataknowsall.com/blog/topicmodels.html
    # Add bigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(texts, min_count=20)
    for idx in range(len(texts)):
        for token in bigram[texts[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                texts[idx].append(token)


def apply_tf_idf(train, test):
    # train and test are cleaned and lemmatized lists
    tf_bigram = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_train_bigram = pd.DataFrame(tf_bigram.fit_transform(train).todense(), columns=tf_bigram.get_feature_names_out())
    X_test_bigram = pd.DataFrame(tf_bigram.transform(test).todense(), columns=tf_bigram.get_feature_names_out())
    return X_train_bigram, X_test_bigram


############################################ Other random utils #######################################################


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
    _SOME_ALPHA_RE = re.compile(r'[A-Za-z]+')
    _ONLY_ALPHA_RE = re.compile(r'^[A-Za-z]*$')

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


def preprocess_steam_review(text):
    # method taken from https://github.com/MullerAC/video-game-review-analysis/
    text = re.sub(r'\[.*?\]', '', text)  # remove markdown tags, only needed for Steam reviews
    text = text.translate(str.maketrans('', '', ''.join(punctuation_list)))  # remove all punctuation
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  # tokenize words with only numbers and latin characters
    return tokenizer.tokenize(text.lower())


def get_most_frequent_words(text_list: list[str]):
    # see https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/calculate-tweet-word
    # -bigrams/
    def remove_url(txt):
        """Replace URLs found in a text string with nothing (i.e. it will remove the URL from the string).
        """
        return " ".join(re.sub(r"([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

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


def textblob_utils(text: str):
    # only for reference
    blob = TextBlob(text)
    print(blob.words)  # split into words
    print(blob.sentences)  # split into sentences
    lemmata = [word.lemmatize() for word in blob.words]
    print(lemmata)
