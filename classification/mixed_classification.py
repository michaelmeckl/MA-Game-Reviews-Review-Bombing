#!/usr/bin/python
# -*- coding: utf-8 -*-
import pprint
import re
import time
import pandas as pd
import torch
import torch.nn as nn
# noinspection PyPep8Naming
from datasets import Dataset as ds
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorWithPadding
from classification import classification_utils
from classification.bert_classifier import BERTClassifier, evaluate_model, train_model, predict_test_labels
from classification.classification_constants import (MODEL_FOLDER, RANDOM_SEED,
                                                     TRAIN_TEST_DATA_FOLDER, PLOT_FOLDER, annotation_questions)
from classification.classification_utils import split_data_scikit, get_pytorch_device, create_test_train_set
from classification.custom_datasets import CustomDataset
from utils import utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def classify_review_bombing(bert_model, train_dataloader: DataLoader, test_dataloader: DataLoader, tag: str,
                            num_epochs=3):
    total_steps = len(train_dataloader) * num_epochs
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=5e-5)  # 3e-5, 2e-5  # see BERT paper for learning rates
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    progress_bar = tqdm(range(total_steps))
    best_loss = 100

    writer = SummaryWriter(f"runs/mixed-{tag}-{ckp_clean}")
    train_history = {
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
        "f1_score": [],
    }

    # load a previous checkpoint to resume training by setting the start_epoch to a different value than 0
    start_epoch = 0
    if start_epoch > 0:
        resume_epoch = start_epoch - 1
        start_epoch = classification_utils.load_model_checkpoint(bert_model, optimizer, MODEL_FOLDER /
                                                                 f"mixed-{tag}-epoch-{resume_epoch}_{ckp_clean}.pt")
        print(f"Resuming training with epoch {start_epoch} ...")

    # train model
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        print(f"##################### Epoch {epoch + 1}/{num_epochs} #####################")
        train_model(bert_model, train_dataloader, optimizer, scheduler, loss_function, device, epoch, writer,
                    train_history, progress_bar)
        val_loss, val_accuracy, report = evaluate_model(bert_model, test_dataloader, loss_function, device, epoch,
                                                        writer,
                                                        train_history)

        # classification_utils.save_model_checkpoint(bert_model, optimizer, epoch,
        #                                            output_path=MODEL_FOLDER / f"mixed-{tag}-epoch-{epoch}_{ckp_clean}.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Best val loss is now: {val_loss:.2f} (val accuracy: {val_accuracy:.2f}%) \n")
            classification_utils.save_model_checkpoint(bert_model, optimizer, epoch,
                                                       output_path=MODEL_FOLDER / f"mixed-{tag}-best-model_{ckp_clean}.pt")
        if epoch == (num_epochs - 1):
            # save the last classification report to file
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(MODEL_FOLDER / f"mixed-report-epoch_{epoch}-{tag}-{ckp_clean}.csv")

    print("Finished training the model!\n")
    finish_time = time.time()
    total_time = finish_time - start_time
    time_per_epoch = total_time / num_epochs
    print(f"Time for training overall: {total_time:.2f} seconds / {(total_time / 60):.2f} min")
    print(f"Time per epoch: {time_per_epoch:.2f} seconds / {(time_per_epoch / 60):.2f} min")

    writer.flush()
    writer.close()
    # plot the training history (loss and accuracy)
    classification_utils.show_training_plot(train_history["train_accuracy"], train_history["val_accuracy"],
                                            train_history["train_loss"], train_history["val_loss"],
                                            train_history["f1_score"],
                                            output_name=f"mixed_train_history_{tag}_{ckp_clean}", show=False)

    return optimizer


def convert_numerical_to_float(data_df: pd.DataFrame, numerical_cols: list[str]):
    # Cast the numerical features to floats.
    for col in numerical_cols:
        data_df[col] = data_df[col].astype('float')
    # or:
    # df[cols] = df[cols].apply(pd.astype('float'), errors='coerce')


def encode_categorical_columns(data_df: pd.DataFrame, cat_cols: list[str]):
    for col in cat_cols:
        data_df[col] = data_df[col].astype('category')
        data_df[col] = data_df[col].cat.codes

    # data_df["Clothing ID"] = data_df["Clothing ID"].astype('category')
    # data_df["Clothing ID"] = data_df["Clothing ID"].cat.codes


def get_feature_combination(row: pd.Series, text_col, option):
    # annotation_questions.remove(target_column)
    game_criticism = "criticism game" if row['criticism-praise-game-content'] == "yes" else "no criticism game"
    dev_criticism = "criticism developer" if row[
                                                 'criticism-praise-developer-publisher'] == "yes" else "no criticism developer"
    personal_criticism = "personal criticism" if row[
                                                     'criticism-praise-ideological'] == "yes" else "no personal criticism"
    political_criticism = "political criticism" if row[
                                                       'criticism-praise-political'] == "yes" else "no political criticism"

    # annotation data
    if option == 1:
        combined = row[text_col]  # review text first
        combined += (f" [SEP] This text contains: {game_criticism}, {dev_criticism}, {personal_criticism}, {political_criticism} ")
        combined += f" [SEP] Related to game" if row['is-rating-game-related'] == "yes" else "Not related to game"
        combined += f" [SEP] Information from {row['num_annotators']} people"
        combined += f" [SEP] Agreement: {row['annotation_certainty']}"
        return combined

    # annotation data without writing additional paragraph
    elif option == 2:
        # only separate with [SEP] tag and dont write paragraph for each? less context but more efficient
        combined = row[text_col]  # review text first
        combined += (f" [SEP] {game_criticism}, {dev_criticism}, {personal_criticism}, {political_criticism} ")
        combined += f" [SEP] Related to game: {row['is-rating-game-related']}"
        combined += f" [SEP] {row['num_annotators']}"
        combined += f" [SEP] {row['annotation_certainty']}"
        return combined

    # social media data
    elif option == 3:
        combined = row[text_col]
        combined += f" [SEP] Sentiment Twitter: {row['avg_sentiment_rb_period - Twitter']}"
        combined += f" [SEP] Sentiment Reddit: {row['avg_sentiment_rb_period - Reddit Posts']}"
        combined += f" [SEP] Relevant Topics Twitter: {row['Topic 0 - Twitter']} {row['Topic 1 - Twitter']} {row['Topic 2 - Twitter']}"
        combined += f" [SEP] Relevant Topics Reddit: {row['Topic 0 - Reddit Posts']} {row['Topic 1 - Reddit Posts']} {row['Topic 2 - Reddit Posts']}"
        return combined
    # only Twitter
    elif option == 9:
        combined = row[text_col]
        combined += f" [SEP] Sentiment Twitter: {row['avg_sentiment_rb_period - Twitter']}"
        combined += f" [SEP] Relevant Topics Twitter: {row['Topic 0 - Twitter']} {row['Topic 1 - Twitter']} {row['Topic 2 - Twitter']}"
        return combined
    elif option == 10:
        combined = row[text_col]
        combined += f" [SEP] Sentiment Reddit: {row['avg_sentiment_rb_period - Reddit Posts']}"
        combined += f" [SEP] Relevant Topics Reddit: {row['Topic 0 - Reddit Posts']} {row['Topic 1 - Reddit Posts']} {row['Topic 2 - Reddit Posts']}"
        return combined

    # combination annotation + social media
    elif option == 4:
        combined = row[text_col]
        combined += (f" [SEP] This text contains: {game_criticism}, {dev_criticism}, {personal_criticism}, {political_criticism} ")
        combined += f" [SEP] Related to game" if row['is-rating-game-related'] == "yes" else "Not related to game"
        combined += f" [SEP] Sentiment Twitter: {row['avg_sentiment_rb_period - Twitter']}"
        combined += f" [SEP] Sentiment Reddit: {row['avg_sentiment_rb_period - Reddit Posts']}"
        combined += f" [SEP] Relevant Topics Twitter: {row['Topic 0 - Twitter']}"
        combined += f" [SEP] Relevant Topics Reddit: {row['Topic 0 - Reddit Posts']}"
        return combined

    # combination annotation + review metadata
    elif option == 5:
        combined = row[text_col]
        combined += (f" [SEP] This text contains: {game_criticism}, {dev_criticism}, {personal_criticism}, {political_criticism} ")
        combined += f" [SEP] Related to game" if row['is-rating-game-related'] == "yes" else "Not related to game"
        combined += f" [SEP] Playtime: {row['author_playtime_at_review_min']}"
        combined += f" [SEP] Author Credibility: {row['author_credibility']}"
        combined += f" [SEP] Game Sentiment: {row['sentiment_score_sentence_level']}"
        combined += f" [SEP] Text is {row['readability_flesch_reading']} to understand"
        return combined

    # review metadata + analysis data
    elif option == 6:
        combined = row[text_col]
        combined += f" [SEP] Playtime: {row['author_playtime_at_review_min']}"
        combined += f" [SEP] Author Credibility: {row['author_credibility']}"
        combined += f" [SEP] Game Sentiment: {row['sentiment_score_sentence_level']}"
        combined += f" [SEP] Text is {row['readability_flesch_reading']} to understand"
        combined += f" [SEP] Relevant Game Topics: {row['Topic 0']} {row['Topic 1']} {row['Topic 2']}"
        return combined

    # review metadata + analysis data without topics
    elif option == 7:
        combined = row[text_col]
        combined += f" [SEP] Playtime: {row['author_playtime_at_review_min']}"
        combined += f" [SEP] Author Credibility: {row['author_credibility']}"
        combined += f" [SEP] Game Sentiment: {row['sentiment_score_sentence_level']}"
        combined += f" [SEP] Text is {row['readability_flesch_reading']} to understand"
        return combined

    # test only with topics from game and social media without review?
    elif option == 8:
        combined = f" [SEP] Relevant Game Topics: {row['Topic 0']} {row['Topic 1']} {row['Topic 2']}"
        combined += f" [SEP] Relevant Topics Twitter: {row['Topic 0 - Twitter']} {row['Topic 1 - Twitter']} {row['Topic 2 - Twitter']}"
        combined += f" [SEP] Relevant Topics Reddit: {row['Topic 0 - Reddit Posts']} {row['Topic 1 - Reddit Posts']} {row['Topic 2 - Reddit Posts']}"
        return combined


def preprocess_combine_all_to_text(df: pd.DataFrame, text_col: str, target_col: str, tokenizer, option=1):
    ######################## split into train and validation set ########################
    # X_data = df[[text_col, "review_bombing_incident", "review_bomb_type", "source"]]
    X_data = df.loc[:, df.columns != target_col]  # select everything but target col
    y_data = df[[target_col]]

    """
    X_data['num_annotators'] = df['num_annotators'].apply(str)
    print(X_data["num_annotators"].dtype)
    X_data['annotation_certainty'] = df['annotation_certainty'].apply(str)
    
    combine_cols = [text_col, *annotation_questions, "num_annotators", "annotation_certainty"]
    X_data["combined_text"] = X_data[[*combine_cols]].fillna('').astype(str).apply(lambda x: " [SEP] ".join(x), axis=1)
    """
    # X_data["combined_text_topics"] = X_data[[text_col, "Topic 0", "Topic 1", "Topic 2", "Topic 3", "Topic 4"]].fillna('').apply(" [SEP] ".join, axis=1)

    #  scale some columns in X_data before converting to text?
    ct = ColumnTransformer(transformers=[
        ("minmax", MinMaxScaler(), ["sentiment_score_sentence_level", "avg_sentiment_rb_period - Twitter", "avg_sentiment_rb_period - Reddit Posts", "author_credibility"]),
        ("standard", StandardScaler(), ["author_playtime_at_review_min"])
    ])
    X_data[["sentiment_score_sentence_level", "avg_sentiment_rb_period - Twitter", "avg_sentiment_rb_period - Reddit Posts", "author_credibility", "author_playtime_at_review_min"]] = ct.fit_transform(X_data)

    sen_w_feats = []
    for index, row in X_data.iterrows():
        combined_text = get_feature_combination(row, text_col, option)
        sen_w_feats.append(combined_text)

    # pprint.pprint(sen_w_feats)
    """
    max_len = 0
    for sent in sen_w_feats:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    print('Max sentence length: ', max_len)
    """

    # add combined text features as a new column to use for training
    X_data.insert(0, "combined_text", sen_w_feats)

    # unfortunately it seems only one additional column can be specified here! "review_bombing_incident" seems to be
    # the best one for the resulting strata
    train_x, val_x, train_y, val_y = split_data_scikit(X_data, y_data, stratify_on=df[[target_column, "review_bombing_incident"]])  # "source"
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    val_x = val_x.reset_index(drop=True)
    val_y = val_y.reset_index(drop=True)

    ############################################
    # tokenize the review column already here instead of in __getitem__ with the tokenize function below, so it does
    # not have to be performed for every batch while training!
    max_tokens = tokenizer.max_model_input_sizes[checkpoint]
    train_dataset = ds.from_pandas(train_x)
    val_dataset = ds.from_pandas(val_x)
    # don't use padding here already, instead a data collator is later used for dynamic padding
    tokenized_train_dataset = train_dataset.map(
        lambda data: tokenizer(data["combined_text"], truncation=True, max_length=max_tokens),
        batched=True)
    tokenized_val_dataset = val_dataset.map(
        lambda data: tokenizer(data["combined_text"], truncation=True, max_length=max_tokens),
        batched=True)
    # tokenized_train_dataset.set_format("torch")  # convert to pytorch dataset
    train_x_tokenized = tokenized_train_dataset.to_pandas()
    val_x_tokenized = tokenized_val_dataset.to_pandas()

    ######################## create custom dataset and dataloader #######################
    # use a data collator to pad the tokens to the longest per batch (see "Dynamic Padding" on https://huggingface.co/learn/nlp-course/en/chapter3/2?fw=pt#dynamic-padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = CustomDataset(train_x_tokenized, train_y)  # transform=tokenize_review
    val_dataset = CustomDataset(val_x_tokenized, val_y)  # transform=tokenize_review
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)  # num_workers = 2
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # test if the batches have the correct shape
    for batch in train_dataloader:
        print({k: v.shape for k, v in batch.items()})
        print(tokenizer.decode(batch['input_ids'][0]))  # show the encoded version of one review
        break
    return train_dataloader, val_dataloader


def train_mixed_model(data: pd.DataFrame, tokenizer, text_col: str, tag: str, option=1):
    print(f"\n[INFO] Testing mixed model option {option}\n")
    # set all the random seeds to make everything reproducible for preprocessing and training
    classification_utils.set_random_seed()
    # random shuffle the data (without it there could be some biases and it would be easier to overfit)
    shuffled_data = data.sample(frac=1, random_state=RANDOM_SEED)

    if use_subset:
        shuffled_data = shuffled_data.head(100)
    print(f"[INFO] Using {len(shuffled_data)} rows/reviews for training ...")

    train_data_loader, val_data_loader = preprocess_combine_all_to_text(shuffled_data, text_col, target_column, tokenizer, option)

    # create the model as well as the training parameters and start training
    num_classes = shuffled_data[target_column].nunique()
    model = BERTClassifier(num_classes, model_checkpoint=checkpoint).to(device)
    classify_review_bombing(model, train_data_loader, val_data_loader, tag)


def predict_on_test_data(test_data: pd.DataFrame, tokenizer, text_col: str, tag: str, incident_positive=False):
    # tokenize
    max_tokens = tokenizer.max_model_input_sizes[checkpoint]
    dataset = ds.from_pandas(test_data[[text_col]])
    test_dataset = dataset.map(lambda data: tokenizer(data[text_col], truncation=True, max_length=max_tokens),
                               batched=True)

    test_x = test_dataset.to_pandas()
    test_y = test_data[[target_column]]
    # setup dataset and data loader
    test_dataset = CustomDataset(test_x, test_y)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator)
    print('\nPredicting labels for {:,} test reviews ...'.format(len(test_dataset)))

    n_classes = test_data[target_column].nunique()
    model = BERTClassifier(n_classes, model_checkpoint=checkpoint).to(device)
    model_checkpoint = torch.load(MODEL_FOLDER / f"mixed-{tag}-best-model_{ckp_clean}.pt")
    model.load_state_dict(model_checkpoint['model_state_dict'])

    tag_for_confusion = f"mixed-{tag}--{ckp_clean}"
    predicted_labels = predict_test_labels(model, test_dataloader, device, tag_for_confusion, incident_positive)
    # show some predictions
    prediction_results = test_data[[text_col, target_column]]
    prediction_results.insert(2, "predictions", predicted_labels)

    # convert 0 and 1 back to categorical label
    encoding = {
        0: "Is Review Bombing" if target_column == "is-review-bombing" else "Is Game-Related",
        1: "Not Review Bombing" if target_column == "is-review-bombing" else "Not Game-Related"
    }
    prediction_results = prediction_results.replace(encoding)
    # for idx, row in prediction_results.iloc[:10].iterrows():
    #     print(f'Review: {row[text_col]}\nActual label: {row[target_column]}\nPredicted label: {row["predictions"]}\n')


if __name__ == "__main__":
    print(f"[INFO] Using PyTorch version: {torch.__version__}")
    utils.enable_max_pandas_display_size()

    if not MODEL_FOLDER.is_dir():
        MODEL_FOLDER.mkdir()
    if not PLOT_FOLDER.is_dir():
        PLOT_FOLDER.mkdir()

    classify_rb = True  # if False classify off_topic column
    target_column = "is-review-bombing" if classify_rb else "is-rating-game-related"
    # text_column = "review"
    text_column = "text_cleaned"

    use_subset = False  # True for testing

    batch_size = 16  # 8

    device = get_pytorch_device()
    # checkpoint = "google-bert/bert-base-uncased"
    # checkpoint = "distilbert-base-uncased"
    checkpoint = "FacebookAI/roberta-base"    # there is no uncased version
    ckp_clean = re.sub("/", "-", checkpoint)  # "clean" version without the / so it can be used in filenames
    print(f"[INFO] Using checkpoint: {checkpoint}")

    create_new_test_train_data = False
    if create_new_test_train_data:
        train_set, test_set = create_test_train_set(cols_to_encode=[target_column])
    else:
        # load existing data
        train_set = pd.read_csv(TRAIN_TEST_DATA_FOLDER / "train_data.csv")
        test_set = pd.read_csv(TRAIN_TEST_DATA_FOLDER / "test_data.csv")
        print(f"Using {len(train_set)} reviews as train set.")
        print(f"Using {len(test_set)} reviews as test set.")

    reviews_to_use = "steam"  # "steam" / "metacritic" / "both"
    if reviews_to_use == "steam":
        print("[INFO] Training model with only steam reviews!\n")
        train_set = train_set[train_set["source"] == "Steam"].reset_index(drop=True)
        model_tag = "steam"
    elif reviews_to_use == "metacritic":
        print("[INFO] Training model with only metacritic reviews!\n")
        train_set = train_set[train_set["source"] == "Metacritic"].reset_index(drop=True)
        model_tag = "metacritic"
    elif reviews_to_use == "both":
        print("[INFO] Training model with steam and metacritic reviews!\n")
        train_set = train_set
        model_tag = "both"
    else:
        raise ValueError("The specified reviews_to_use type is unknown!")

    CURRENT_OPTION = 0
    for option in [5]:
        CURRENT_OPTION = option
        # update the tag for the model save
        option_tag = f"option-{CURRENT_OPTION}-{model_tag}_review_bombing" if classify_rb else f"{model_tag}_game_related"

        pre_trained_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        train_mixed_model(train_set, pre_trained_tokenizer, text_column, option_tag, option=option)

        # predict on positive and on negative test incident
        test_set_positive = test_set[test_set["review_bomb_type"] == "positiv"]
        test_set_negative = test_set[test_set["review_bomb_type"] == "negativ"]
        test_set_positive = test_set_positive.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        test_set_negative = test_set_negative.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        predict_on_test_data(test_set_positive, pre_trained_tokenizer, text_column, option_tag, incident_positive=True)
        predict_on_test_data(test_set_negative, pre_trained_tokenizer, text_column, option_tag)

        print("#########################################################################################\n")