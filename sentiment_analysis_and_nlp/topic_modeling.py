#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
import pprint
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore, LdaModel
from joblib import dump, load
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
# noinspection PyPep8Naming
from sklearn.decomposition import LatentDirichletAllocation as LDA, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm
from sentiment_analysis_and_nlp.nlp_utils import apply_standard_text_preprocessing
from sentiment_analysis_and_nlp.spacy_utils import SpacyUtils
from utils.utils import enable_max_pandas_display_size


def save_load_trained_model(topic_model, model_name="default", output_path=".", save=True):
    """
    Used to save trained topic model, e.g. from scikit-learn. Gensim provides its own save mechanics simply with
    "model.save()" and loaded with "models.LdaModel.load()".
    """
    if save:
        dump(topic_model, f'{output_path}/topic_model_{model_name}.joblib')
    else:
        return load(f'{output_path}/topic_model_{model_name}.joblib')


############################################# Gensim #############################################


def compute_coherence_values(dictionary, corpus, texts, cohere, limit, chunksize, iterations, passes, eval_every,
                             start=2, step=2):
    # Taken from https://dataknowsall.com/blog/topicmodels.html
    coherence_values = []
    for num_topics in range(start, limit, step):
        # use LdaMulticore() instead of LdaModel() to speed up training ? faster but some claim it results in
        # worse topics ?
        """
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, chunksize=chunksize, alpha='auto',
                         eta='auto', iterations=iterations, passes=passes, eval_every=eval_every, random_state=42,
                         per_word_topics=False)
        """
        # TODO leave workers out if something does not work or set to lower number
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, chunksize=chunksize,
                             iterations=iterations, passes=passes, eval_every=eval_every, random_state=42,
                             per_word_topics=False, workers=7)

        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence=cohere)
        coherence_values.append(coherencemodel.get_coherence())

    return coherence_values


def perform_gensim_lda(text_list: list[str]):
    # Taken from https://dataknowsall.com/blog/topicmodels.html
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    tokenized_text_data = [nltk.tokenize.word_tokenize(doc) for doc in text_list]

    # Create the dictionary
    dictionary = corpora.Dictionary(tokenized_text_data)
    # Filter out words that occur less than X documents or more than X% of the documents.
    dictionary.filter_extremes(no_below=5, no_above=0.6)

    # Create the corpus. This is a Term Frequency or Bag of Words representation.
    corpus = [dictionary.doc2bow(text) for text in tokenized_text_data]

    print(f'Number of unique tokens: {len(dictionary)}')
    print(f'Number of documents: {len(corpus)}')

    chunksize = 2000
    passes = 6   # 10
    iterations = 100
    eval_every = 1

    # Try to find the best number of topics for the given corpus
    limit = 20   # 50
    start = 2
    step = 2
    coherence_values = compute_coherence_values(dictionary, corpus, text_list, cohere='c_v', limit=limit,
                                                chunksize=chunksize, iterations=iterations, passes=passes,
                                                eval_every=eval_every, start=start, step=step)
    # plot the coherence values
    plt.figure(figsize=(8, 5))
    # Create a custom x-axis for the number of topics
    x = range(start, limit, step)

    ax = sns.lineplot(x=x, y=coherence_values, color='#238C8C')
    plt.title("Best Number of Topics for LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.xlim(start, limit)
    plt.xticks(range(2, limit, step))
    # Add a vertical line to show the optimum number of topics
    # plt.axvline(x[np.argmax(coherence_values)], color='#F26457', linestyle='--')

    # Draw a custom legend
    legend_elements = [Line2D([0], [0], color='#238C8C',
                              ls='-', label='Coherence Value (c_v)'),
                       Line2D([0], [1], color='#F26457',
                              ls='--', label='Optimal Number of Topics')]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

    """
    # train again with the best model and show the topics
    id2word = dictionary.id2token
    n_topics = 6
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=n_topics,
                         random_state=100,
                         update_every=1,
                         chunksize=10,
                         passes=10,
                         alpha='symmetric',
                         iterations=100,
                         per_word_topics=True)
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} Word: {}".format(idx, topic))
        print("\n")
    
    # visualize topics
    import pyLDAvis.gensim_models
    lda_viz = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary=lda_model.id2word, sort_topics=True)
    pyLDAvis.display(lda_viz)
    """


"""
def perform_gensim_lsi(input_data: list[str]):
    # Code based in parts on https://github.com/alfredtangsw/steamvox/tree/master
    tokenized_text_data = [nltk.tokenize.word_tokenize(doc) for doc in input_data]
    dictionary = corpora.Dictionary(tokenized_text_data)
    # Filter out words that occur less than X documents or more than X% of the documents.
    dictionary.filter_extremes(no_below=10, no_above=0.6)

    corpus = [dictionary.doc2bow(text) for text in tokenized_text_data]

    use_lsi = True
    if use_lsi:
        # Train Latent Semantic Indexing with 300D vectors.
        num_topics = 300
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
        corpus_model = model[corpus_tfidf]
"""


############################################# Scikit-Learn #############################################

def perform_sklearn_lda_nmf(df: pd.DataFrame, text_col: str):
    # Code based on https://medium.com/@sehjadkhoja0/title-exploring-sentiment-analysis-and-topic-modeling-with-daily-tweets-4b4c5db39013
    use_tfidf = True
    if use_tfidf:
        vectorizer = TfidfVectorizer(min_df=0.05, ngram_range=(1, 3))
        dtm = vectorizer.fit_transform(df[text_col])
    else:
        # Create a document-term matrix using CountVectorizer
        vectorizer = CountVectorizer(stop_words="english", max_features=5000)
        dtm = vectorizer.fit_transform(df[text_col])

    num_topics = 10  # test different values

    # Fit LDA model
    lda_model = LDA(n_components=num_topics, random_state=42)
    lda_model.fit(dtm)

    # Fit NMF model
    nmf_model = NMF(n_components=num_topics, init="nndsvd", random_state=42)
    nmf_model.fit(dtm)

    # Get the topic-word distributions for LDA and NMF
    lda_topic_word_distributions = lda_model.components_
    nmf_topic_word_distributions = nmf_model.components_

    # Get the document-topic distributions for LDA and NMF
    lda_document_topic_distributions = lda_model.transform(dtm)
    nmf_document_topic_distributions = nmf_model.transform(dtm)

    # Transform the document-term matrix using the fitted model
    # For LDA
    lda_topic_documents = lda_model.transform(dtm).argmax(axis=1)
    most_representative_documents_lda = []
    for topic_id in range(lda_model.n_components):
        documents_df = df.iloc[lda_topic_documents == topic_id]
        # documents_index = df.iloc[lda_topic_documents == topic_id].index
        documents = documents_df[text_col].to_list()
        most_representative_documents_lda.extend(documents)

    # For NMF
    nmf_topic_documents = nmf_model.transform(dtm).argmax(axis=1)
    most_representative_documents_nmf = []
    for topic_id in range(nmf_model.n_components):
        documents_df = df.iloc[nmf_topic_documents == topic_id]
        documents = documents_df[text_col].to_list()
        most_representative_documents_nmf.extend(documents)

    # Get the top 5 terms for each topic
    top_n_terms = 5
    print("\nLDA Topics:")
    for topic_id, topic in enumerate(lda_model.components_):
        top_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()][::-1][:top_n_terms]
        print(f"Topic {topic_id + 1}: {top_terms}")

    get_coherence_score(df, text_col, most_representative_documents_lda)

    print("\nNMF Topics:")
    for topic_id, topic in enumerate(nmf_model.components_):
        top_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()][::-1][:top_n_terms]
        print(f"Topic {topic_id + 1}: {top_terms}")

    get_coherence_score(df, text_col, most_representative_documents_nmf)


def get_coherence_score(df: pd.DataFrame, text_col: str, most_representative_documents):
    # Code based on https://medium.com/@sehjadkhoja0/title-exploring-sentiment-analysis-and-topic-modeling-with-daily-tweets-4b4c5db39013
    # Apply this after the sklearn topic modeling above
    tokenized_text_col = df[text_col].apply(lambda x: nltk.tokenize.word_tokenize(x))
    dictionary = corpora.Dictionary(tokenized_text_col)

    tokenized_documents = [document for document in most_representative_documents]
    # Tokenize the documents for each topic
    # tokenized_documents = [nltk.tokenize.word_tokenize(document) for document in most_representative_documents]

    # Calculate the coherence score using the c_v coherence measure
    # The coherence score measures the semantic coherence and interpretability of the topics by examining the relationships between words within each topic
    coherence_model = CoherenceModel(
        topics=tokenized_documents,
        texts=df[text_col],
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    print("Topic Coherence Score: {:.4f}".format(coherence_score))


############################################# BERTopic #############################################

def perform_topic_modeling_bertopic(input_data: list[str], tag: str, use_spacy_embedding=False,
                                    use_custom_vectorizer=False, use_custom_tf_idf=False, is_reddit=False):
    # (1, 2) to use unigrams and bigrams; (1, 3) to use unigrams, bigrams and trigrams
    n_gram_range = (1, 3)

    # specify embedding model instead of using sentence transformers as per default
    if is_reddit:
        # since reddit posts and comments are usually pretty long, they need to be split up in sentences first
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # sent_embeddings = embedding_model.encode(sentences, show_progress_bar=True)

        batch_size = 16
        n = len(input_data)
        sent_embeddings = np.zeros((n, embedding_model.get_sentence_embedding_dimension()))

        for i in tqdm(range(0, n, batch_size)):
            i_end = min(i + batch_size, n)
            batch = input_data[i:i_end]
            batch_embed = embedding_model.encode(batch)
            sent_embeddings[i:i_end, :] = batch_embed
    else:
        embedding_model = SpacyUtils().spacy_nlp if use_spacy_embedding else None
        print(f"Using spacy: {embedding_model}")
        sent_embeddings = None

    # with the CountVectorizer it is possible to remove stop words for topic modeling AFTER having generated
    # embeddings and clustered the documents, see https://maartengr.github.io/BERTopic/faq.html#how-do-i-remove-stop-words
    # -> in theory this should be better than removing them before (at least if spacy embeddings are not used)
    custom_vectorizer_model = CountVectorizer(stop_words="english", min_df=2,
                                              ngram_range=n_gram_range) if use_custom_vectorizer else None
    custom_ctfidf_model = ClassTfidfTransformer(bm25_weighting=False,
                                                reduce_frequent_words=True) if use_custom_tf_idf else None
    print(f"Using custom vectorizer: {custom_vectorizer_model}")
    print(f"Using custom c-tf-idf transformer: {custom_ctfidf_model}")

    # update the topic representation
    main_representation = KeyBERTInspired()
    aspect_representation = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=0.5)]
    custom_representation_model = {
        "main": main_representation,
        "aspect": aspect_representation
    }

    topic_model = BERTopic(embedding_model=embedding_model, top_n_words=10, n_gram_range=n_gram_range,
                           verbose=False, nr_topics="auto", vectorizer_model=custom_vectorizer_model,
                           ctfidf_model=custom_ctfidf_model,     # calculate_probabilities=True, min_topic_size=12
                           representation_model=custom_representation_model)
    topics, probs = topic_model.fit_transform(input_data, embeddings=sent_embeddings)

    def print_modeling_results(model, run=""):
        print("#############\n")
        # Topic -1 declares outliers that could not be assigned to any topic
        topic_info_df = model.get_topic_info()
        pprint.pprint(topic_info_df)
        topic_info_df.to_csv(OUTPUT_FOLDER / f"bertopic_topic_info_{run}_{tag}.csv", index=False)
        # print("#############\n")
        # document_info_df = model.get_document_info(input_data)
        # pprint.pprint(document_info_df)
        # show infos for the most frequent topic
        # pprint.pprint(model.get_topic(0))
        print("#############\n")
        print("Topic labels and representative words:")
        pprint.pprint(model.topic_labels_)
        print("#############\n")

    def visualize_modeling_results(model, run=""):
        # replace default topic names ("Topic 1") with custom labels
        topic_words = model.generate_topic_labels(nr_words=2, separator=" - ")  # , word_length=15
        model.set_topic_labels(topic_words)

        # visualize the topics with plotly
        """
        try:
            fig_1 = model.visualize_topics(title=f"<b>Intertopic Distance Map - {run} {tag}</b>")
            fig_1.show()
            fig_1.write_html(OUTPUT_FOLDER / f"Intertopic Distance Map - {run} {tag}.html")
        except Exception as e:
            print(f"[ERROR] plot 1: {e}")
        """
        try:
            fig_2 = model.visualize_barchart(title=f"Topic Words - {run} {tag}", top_n_topics=10,
                                             width=450, height=300, custom_labels=True)
            fig_2.show()
            fig_2.write_html(OUTPUT_FOLDER / f"Topic Words - {run} {tag}.html")
        except Exception as e:
            print(f"[ERROR] plot 2: {e}")

    # show the results of the first run
    # print_modeling_results(topic_model)
    # visualize_modeling_results(topic_model)

    update_model = True
    if update_model:
        new_ngram_range = (2, 3)
        """
        custom_vectorizer_model = CountVectorizer(stop_words="english") if use_custom_vectorizer else None
        custom_ctfidf_model = ClassTfidfTransformer(bm25_weighting=True,
                                                    reduce_frequent_words=True) if use_custom_tf_idf else None
        """
        # update model with new_ngram_range
        topic_model.update_topics(input_data, n_gram_range=new_ngram_range)

        # show the results of the updated run
        print_modeling_results(topic_model, run="second run")
        visualize_modeling_results(topic_model, run="Second run")

    # save the (updated) trained model
    topic_model.save(OUTPUT_FOLDER / f"bertopic_{tag}", serialization="safetensors", save_ctfidf=True)


def apply_trained_bertopic_model(input_text: str, tag: str):
    # load trained model
    loaded_topic_model = BERTopic.load(OUTPUT_FOLDER / f"bertopic_{tag}")
    topics, probs = loaded_topic_model.transform(input_text)
    assigned_topic = loaded_topic_model.topic_labels_[topics[0]]   # get the topic result for the input
    print(f"\nAssigned topic: {assigned_topic}")
    # get the most likely topics from the trained model for this input
    print(f"Suggested topics:")
    similar_topics, similarity = loaded_topic_model.find_topics(assigned_topic)
    for sim_topic_id in similar_topics:
        print(loaded_topic_model.topic_labels_[sim_topic_id])


def start_topic_modeling(use_option=2):
    INPUT_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "posts"
    rb_incidents = ["Skyrim-Paid-Mods", "Assassins-Creed-Unity", "Firewatch", "Mortal-Kombat-11",
                    "Borderlands-Epic-Exclusivity", "Ukraine-Russia-Conflict"]
    rb_incidents = ["Assassins-Creed-Unity"]   # TODO testing

    for rb_name in rb_incidents:
        incident_folder = INPUT_FOLDER / rb_name
        twitter_data = pd.read_csv(incident_folder / f"twitter_combined_{rb_name}.csv",
                                   nrows=100)   # TODO use all later
        reddit_submission_data = pd.read_csv(incident_folder / f"reddit_submissions_combined_{rb_name}.csv",
                                             nrows=500)
        reddit_comment_data = pd.read_csv(incident_folder / f"combined_reddit_comments_{rb_name}.csv",
                                          nrows=500)

        # TODO also use reddit comments and submissions! add "combined_content" to the twitter list?

        # test multilingual topic modeling too ? (e.g. for bertopic set language parameter to multilingual)
        english_data = reddit_submission_data[reddit_submission_data["detected_language"] == "english"]
        print(f"Using {len(english_data)} documents for topic modeling for incident {rb_name}...")

        # Todo also only use in rb time period?
        # posts_in_rb_time = english_data[english_data["in_rb_time_period"]]

        # remove stop words, clean text and lemmatize
        apply_standard_text_preprocessing(english_data, text_col="combined_content", is_social_media_data=True)

        docs_original = list(english_data["combined_content"])
        docs_cleaned = list(english_data["text_cleaned"])  # to lowercase and special characters + stopwords removed
        docs_lemmatized = list(english_data["text_preprocessed"])  # docs_cleaned + lemmatization

        is_reddit_data = False
        source_tag = ""
        if english_data["source"].iloc[0] == "Twitter":
            source_tag = "tw_"
        elif english_data["source"].iloc[0] == "Reddit":
            source_tag = "red_"
            is_reddit_data = True

        match use_option:
            case 0:
                print("Using Sklearn LDA ...")
                perform_sklearn_lda_nmf(english_data, text_col="text_preprocessed")
            case 1:
                print("Using Gensim ...")
                perform_gensim_lda(docs_lemmatized)
            case 2:
                print("Using BERTopic ...")
                if is_reddit_data:
                    print("\nTraining on sentence-level for reddit ....")
                    # split into sentences first
                    sentence_list = [sent_tokenize(post) for post in docs_cleaned]    # docs_original
                    sentences = [sentence for doc in sentence_list for sentence in doc]

                    perform_topic_modeling_bertopic(sentences, tag=f"{source_tag}cleaned_custom_tfidf_{rb_name}",
                                                    use_custom_tf_idf=True, is_reddit=is_reddit_data)
                else:
                    print("\nTraining with not-preprocessed input data ....")
                    # perform_topic_modeling_bertopic(docs_original, tag=f"{source_tag}original_custom_vectorizer_{rb_name}",
                    #                                use_custom_vectorizer=True, is_reddit=is_reddit_data)
                    perform_topic_modeling_bertopic(docs_original, tag=f"{source_tag}original_custom_tfidf_{rb_name}",
                                                    use_custom_tf_idf=True, is_reddit=is_reddit_data)
                    print("\nTraining with cleaned input data ....")
                    perform_topic_modeling_bertopic(docs_cleaned, tag=f"{source_tag}cleaned_custom_tfidf_{rb_name}",
                                                    use_custom_tf_idf=True, is_reddit=is_reddit_data)
                    # print("\nTraining with cleaned and lemmatized input data ....")
                    # perform_topic_modeling_bertopic(docs_lemmatized, tag=f"{source_tag}lemmatized_{rb_name}", is_reddit=is_reddit_data)
            case _:
                print("No suitable option found!")


if __name__ == "__main__":
    enable_max_pandas_display_size()
    OUTPUT_FOLDER = pathlib.Path(__file__).parent / "trained_topic_models"

    if not OUTPUT_FOLDER.is_dir():
        OUTPUT_FOLDER.mkdir()

    start_topic_modeling(use_option=2)
    # apply_trained_bertopic_model("thx ubi for notre dame", tag="original_custom_vec_Assassins-Creed-Unity")

    # TODO bring exported topic modeling into a format where they can be used for training as additional metadata (how?)

    # TODO maybe train topic model on all tweets, submissions and comments and then apply it to the reviews? does
    #  this work to assign them to suitable topics? can this applied topic be used as an additional column for each
    #  review for training?
