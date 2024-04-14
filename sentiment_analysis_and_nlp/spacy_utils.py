import spacy


class SpacyUtils:
    def __init__(self):
        # spacy.download("en_core_web_sm")
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def get_stopwords(self):
        return self.spacy_nlp.Defaults.stop_words

    def split_into_words(self, text: str, remove_stop=False):
        doc = self.spacy_nlp(text)
        if remove_stop:
            token_list = [token for token in doc if not token.is_stop]
        else:
            token_list = [token for token in doc]
        return token_list

    def split_into_sentences(self, text: str):
        doc = self.spacy_nlp(text)
        sentence_list = [sentence for sentence in doc.sents]
        return sentence_list

    def english_tokenize(self, text):
        """
        Tokenize an English text and return a list of tokens with spacy
        """
        return [token.text for token in self.spacy_nlp.tokenizer(text)]

    def lemmatize_text(self, text):
        nlp_text = self.spacy_nlp(text)
        lemmatized_tokens = [y.lemma_ for y in nlp_text]   # lemma_.strip().lower()
        lemmatized_text = " ".join(lemmatized_tokens)
        return lemmatized_tokens, lemmatized_text

    def named_entity_recognition(self, input_text: str):
        nlp_text = self.spacy_nlp(input_text)

        def is_token_allowed(token):
            if not token or not token.string.strip() or token.is_stop or token.ispunct:
                return False
            return True

        # lemmatize the text with spacy first
        complete_filtered_tokens = [
            token.lemma_.strip().lower() for token in nlp_text if is_token_allowed(token)
        ]
        lemmatized_text = " ".join(complete_filtered_tokens)
        print(f"Lemmatized Text: {lemmatized_text}")

        # named entity recognition
        doc = self.spacy_nlp(lemmatized_text)

        # Extract named entities and their relationships
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        relations = [(entity.text, entity.root.dep_, entity.root.head.text) for entity in doc.ents]
        print("Named Entities:", entities)
        print("Relations:", relations)
