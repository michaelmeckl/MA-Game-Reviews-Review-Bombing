#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
import string
import pprint
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import evaluate
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from datasets import Dataset
from classification.classification_utils import split_data_scikit, encode_target_variable, split_data_pytorch
from classification.classifier import BERTClassifier, predict_label, evaluate_model, train_model
from classification.custom_datasets import CustomBaselineDataset, BertCustomDataset, CustomDataset
from sentiment_analysis_and_nlp import nltk_utils
from sentiment_analysis_and_nlp.spacy_utils import SpacyUtils
from utils import utils
from classification import classification_utils


INPUT_DATA_FOLDER = pathlib.Path(__file__).parent.parent / "label_studio_study" / "parsed_data"


# similar method to how tokenization in preparing for label studio was done:
def tokenize_input(dataframe: pd.DataFrame):
    dataset = Dataset.from_pandas(dataframe)
    tokenized_dataset = dataset.map(lambda df: tokenizer(df["review"], padding=True, truncation=True), batched=True)
    return tokenized_dataset


def preprocess_categorical_data(df: pd.DataFrame, column_names: list[str], use_word_split=True, use_spacy=True):
    """
    1) Tokenizing sentences to break text down into sentences, words, or other units such as tokens
    2) Removing stop words
    3) Normalizing words by condensing all forms of a word into a single form (i.e. stemming or lemmatization)
    4) Vectorizing text by turning the text into a numerical representation
    """
    data_test = df["review"][0]

    if use_spacy:
        spacy_utils = SpacyUtils()
        if use_word_split:
            tokens, lemmata = spacy_utils.split_into_words(data_test)
            vector_representation = tokens[1].vector
        else:
            # else use tokenizer
            tokens = spacy_utils.english_tokenize(data_test)
            # df["review"] = df["review"].apply(english_tokenize)
            # todo stopword removal and lemmatization still missing here ? check tokens!

    else:
        # TODO remove punctuation first and replace it with a whitespace or keep it?
        review_without_punct = [" " if char in string.punctuation else char for char in data_test]
        review_without_punct = "".join(review_without_punct)
        # then split into words, remove stopwords and lemmatize
        tokenized_text = nltk_utils.split_into_words(review_without_punct)
        tokenized_text_without_stop = nltk_utils.remove_stopwords(tokenized_text)
        lemmatized_text = nltk_utils.lemmatize_text(tokenized_text_without_stop)
        # TODO vectorization still missing here

    return df


def classify_review_bombing(df: pd.DataFrame):
    pass


def preprocess_data_version_2(df: pd.DataFrame):
    relevant_data = df.filter(["review", *annotation_questions])
    print("==============================================")
    print(f'The shape of the relevant_data is: {relevant_data.shape}')
    print("==============================================")
    print(f'The number of values for "is-review-bombing" is:\n{relevant_data["is-review-bombing"].value_counts()}')
    print("==============================================")

    # encode the target variables
    encode_target_variable(relevant_data, annotation_questions, use_label_encoder=False)

    # split into train and test data
    train_data, test_data = split_data_pytorch(relevant_data)

    # create dataset and dataloader
    batch_size = 32
    training_dataset = CustomBaselineDataset(train_data, num_labels=2)
    test_dataset = CustomBaselineDataset(test_data, num_labels=2)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def preprocess_data_version_1(df: pd.DataFrame):
    ###################### encode categorical variables #####################
    encode_target_variable(df, annotation_questions)

    ######################## split into train and test set ########################
    X_data = df[['review']]
    y_data = df[['is-review-bombing']]

    # TODO split off a test set and save it as a separate csv file to make sure none of the classifiers will see it
    #  while training! and to make sure it's the same for all! (which reviews should be used as the test set ?)
    train_x, test_x, train_y, test_y = split_data_scikit(X_data, y_data)
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    ######################## create custom dataset and dataloader #######################
    def tokenize_review(review: str):
        max_length = tokenizer.max_model_input_sizes[checkpoint]
        encoding = tokenizer(review, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        # input_ids = encoding['input_ids'][0]
        # attention_mask = encoding['attention_mask'][0]
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        # resize the tensors to the same size
        input_ids = nn.functional.pad(input_ids, (0, max_length - input_ids.shape[0]), value=0)
        attention_mask = nn.functional.pad(attention_mask, (0, max_length - attention_mask.shape[0]), value=0)
        return input_ids, attention_mask

    batch_size = 32
    train_dataset = CustomDataset(train_x, train_y, transform=tokenize_review)
    test_dataset = CustomDataset(test_x, test_y, transform=tokenize_review)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    print(f"Using PyTorch version: {torch.__version__}")
    utils.enable_max_pandas_display_size()
    classification_utils.set_random_seed()

    annotation_questions = ["is-review-bombing", "is-rating-game-related", "criticism-praise-game-content",
                            "criticism-praise-developer-publisher", "criticism-praise-ideological",
                            "criticism-praise-political"]
    # load relevant data
    # TODO for testing
    num_rows = 10
    combined_annotated_data = pd.read_csv(INPUT_DATA_FOLDER / "combined_final_annotation_all_projects.csv",
                                          nrows=num_rows)
    # random shuffle the data
    combined_annotated_data = combined_annotated_data.sample(frac=1)  # , random_state=42

    """ Vorgehen:
    1) Tokenisierung und Vektorisierung:
        Tokenisieren Sie den Text, um ihn in einzelne Wörter oder Sätze aufzuteilen.
        Verwenden Sie Techniken wie Word Embeddings (z.B. Word2Vec, GloVe) oder Transformer-Modelle (z.B. BERT, GPT) zur Vektorisierung des Textes.
    2) Padding und Sequenzierung:
        Fügen Sie Padding hinzu, um alle Texte auf die gleiche Länge zu bringen.
        Sequenzieren Sie die Token, um dem Modell die Reihenfolge der Wörter im Text mitzuteilen.
    3) Embedding-Schicht im Deep Learning-Modell:
        Integrieren Sie eine Embedding-Schicht am Anfang des Deep Learning-Modells, um die tokenisierten und vektorisierten Textdaten als Eingabe zu akzeptieren.
    """

    # create tokenizer and preprocess the data
    checkpoint = "google-bert/bert-base-cased"
    # bert_tokenizer = BertTokenizer.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  # uses a Rust-based fast tokenizer version instead of Python
    train_data_loader, test_data_loader = preprocess_data_version_1(combined_annotated_data)

    # create the model and the training parameters
    learning_rate = 2e-5
    num_epochs = 2
    num_classes = combined_annotated_data["is-review-bombing"].nunique()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = BERTClassifier(num_classes, model_checkpoint=checkpoint).to(device)

    optimizer = AdamW(bert_model.parameters(), lr=learning_rate)
    total_steps = len(train_data_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # train model
    for epoch in range(num_epochs):
        print(f"##################### Epoch {epoch + 1}/{num_epochs} #####################")
        train_model(bert_model, train_data_loader, optimizer, scheduler, device)
        accuracy, report = evaluate_model(bert_model, test_data_loader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

    # save model locally
    # torch.save(bert_model.state_dict(), 'baseline_model.pt')   # 'baseline_model.pth'

    # test prediction
    test_review = "The game was great and I really enjoyed the combat and the story."
    predicted_label = predict_label(test_review, bert_model, tokenizer, device)
    print(test_review)
    print(f"Predicted label: {predicted_label}")

    """
    # TODO model does not accept two input for input_ids and attention_mask
    model = nn.Sequential(
        nn.Linear(60, 30),
        nn.ReLU(),
        nn.Linear(30, 2),
        nn.Sigmoid()
    )
    # Train the model
    n_epochs = 2  # 200
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in data_loader:
            y_pred = model(X_batch)
            # TODO y_pred.values.ravel() ? or y_pred.ravel() ?
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # create one test tensor from the testset
    X_test, y_test = default_collate((x_test.to_numpy()[0], y_test.to_numpy()[0]))
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc)
    print("Model accuracy: %.2f%%" % (acc * 100))
    """

    ##############
    """
    train_dataset = BertCustomDataset(train_data, tokenizer)
    test_dataset = BertCustomDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_ids, attention_mask, labels_tensor = next(iter(train_loader))
    """

    ############## alternative:
    """
    training_data = CustomBaselineDataset(annotated_data_train, num_labels=2)
    test_data = CustomBaselineDataset(annotated_data_test, num_labels=2)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Display review and label
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    review_text = train_features[0].squeeze()
    review_label = train_labels[0]
    print(f"Review:\n   \"{review_text}\"")
    print(f"Label: {review_label}\n")
    """

    ###########################################################################
