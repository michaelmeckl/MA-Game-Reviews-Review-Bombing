import spacy
import spacy_fastlang  # noqa: F401 # pylint: disable=unused-import
from langdetect import detect, LangDetectException, DetectorFactory, detect_langs


# make langdetect deterministic, see https://pypi.org/project/langdetect/
DetectorFactory.seed = 0


def detect_language(x):
    try:
        return detect(x)
    except (LangDetectException, TypeError):
        return 'unknown'


def detect_contains_english(x):
    try:
        languages = detect_langs(x)
        # print(f"\nDetected languages \"{languages}\" for text: \"{x}\"")
        for lang_el in languages:
            if lang_el.lang == "en":
                return True
        return False
    except (LangDetectException, TypeError):
        return False


def setup_spacy_language_detection():
    # setup spacy language detector
    spacy_en = spacy.load("en_core_web_sm")
    spacy_en.add_pipe("language_detector")
    return spacy_en


# noinspection PyProtectedMember
def detect_language_spacy(x, spacy_nlp, min_probability_score=0.3):
    # a bit less restrictive than lang_detect above but where lang_detect removes too much, spacy_fastlang includes
    # some obvious non-english texts as well ...
    try:
        doc = spacy_nlp(x)
        detected_language = doc._.language
        # make sure the probability score is high enough for the found english reviews
        if detected_language == "en":
            if doc._.language_score < min_probability_score:
                return "unknown"
        return detected_language
    except Exception as e:
        print(f"Error while trying to detect language with Spacy: {e}")
        return 'unknown'
