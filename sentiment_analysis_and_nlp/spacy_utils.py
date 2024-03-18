import spacy


class SpacyUtils:
    def __init__(self):
        # spacy.download("en_core_web_sm")
        self.spacy_nlp = spacy.load("en_core_web_sm")
        # self.spacy_stopwords = self.spacy_nlp.Defaults.stop_words

    def split_into_words(self, text: str, remove_stop=True):
        doc = self.spacy_nlp(text)
        if remove_stop:
            token_list = [token for token in doc if not token.is_stop]
        else:
            token_list = [token for token in doc]
        lemmatized_token_list = [token.lemma_ for token in token_list]
        return token_list, lemmatized_token_list

    def split_into_sentences(self, text: str):
        doc = self.spacy_nlp(text)
        sentence_list = [sentence for sentence in doc.sents]
        return sentence_list

    def english_tokenize(self, text):
        """
        Tokenize an English text and return a list of tokens with spacy
        """
        return [token.text for token in self.spacy_nlp.tokenizer(text)]

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
        lemmatized_sentence = " ".join(complete_filtered_tokens)
        print(f"Lemmatized Sentence: {lemmatized_sentence}")

        # named entity recognition TODO find out which others exist and which are useful for me
        nlp_text = self.spacy_nlp(lemmatized_sentence)
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
