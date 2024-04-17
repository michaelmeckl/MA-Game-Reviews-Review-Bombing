#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import time
import pandas as pd
import torch
import torch.nn as nn
# noinspection PyPep8Naming
from datasets import Dataset as ds
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorWithPadding
from classification import classification_utils
from classification.classification_constants import MODEL_FOLDER, INPUT_DATA_FOLDER, annotation_questions
from classification.classification_utils import split_data_scikit, encode_target_variable, split_data_pytorch, \
    get_pytorch_device
from classification.classifier import BERTClassifier, predict_label, evaluate_model, train_model, \
    get_pretrained_bert_for_sequence
from classification.custom_datasets import CustomBaselineDataset, CustomDataset
from sentiment_analysis_and_nlp.nlp_utils import apply_standard_text_preprocessing
from utils import utils


def classify_review_bombing(bert_model, train_dataloader: DataLoader, test_dataloader: DataLoader, tag: str,
                            num_epochs=3):
    total_steps = len(train_dataloader) * num_epochs
    # TODO use nn.BCEWithLogitsLoss() instead? -> switch num_classes to 1 and adjust loss calculation in train/eval loop
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)  # 3e-5, 5e-5  # see BERT paper for learning rates
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    progress_bar = tqdm(range(total_steps))
    best_loss = 100

    writer = SummaryWriter(f"runs/baseline-{tag}-{ckp_clean}")
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
                                                                 f"baseline-{tag}-epoch-{resume_epoch}_{ckp_clean}.pt")
        print(f"Resuming training with epoch {start_epoch} ...")

    # train model
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        print(f"##################### Epoch {epoch + 1}/{num_epochs} #####################")
        train_model(bert_model, train_dataloader, optimizer, scheduler, loss_function, device, epoch, writer,
                    train_history, progress_bar)
        val_loss, val_accuracy = evaluate_model(bert_model, test_dataloader, loss_function, device, epoch, writer,
                                                train_history)

        classification_utils.save_model_checkpoint(bert_model, optimizer, epoch,
                                                   output_path=MODEL_FOLDER / f"baseline-{tag}-epoch-{epoch}_{ckp_clean}.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Best val loss is now: {val_loss:.2f} (val accuracy: {val_accuracy:.2f}%) \n")
            classification_utils.save_model_checkpoint(bert_model, optimizer, epoch,
                                                       output_path=MODEL_FOLDER / f"baseline-{tag}-best-model_{ckp_clean}.pt")
    print("Finished training the model!\n")
    finish_time = time.time()
    total_time = finish_time - start_time
    time_per_epoch = total_time / num_epochs
    print(f"Time for training overall: {total_time:.2f} seconds / {(total_time/60):.2f} min")
    print(f"Time per epoch: {time_per_epoch:.2f} seconds / {(time_per_epoch/60):.2f} min")

    writer.flush()
    writer.close()
    # plot the training history (loss and accuracy)
    classification_utils.show_training_plot(train_history["train_accuracy"], train_history["val_accuracy"],
                                            train_history["train_loss"], train_history["val_loss"], train_history["f1_score"],
                                            output_folder=MODEL_FOLDER, output_name=f"train_history_{tag}_{ckp_clean}",
                                            show=False)

    return optimizer


def preprocess_data_version_2(df: pd.DataFrame, text_col: str, target_col: str):
    batch_size = 16
    relevant_columns = df.filter([text_col, *annotation_questions])
    print("==============================================")
    print(f'The shape of the relevant columns is: {relevant_columns.shape}')
    print("==============================================")
    print(f'The number of values for "{target_col}" is:\n{relevant_columns[target_col].value_counts()}')
    print("==============================================")

    # encode the target variables
    encode_target_variable(relevant_columns, target_col, annotation_questions, use_label_encoder=False)

    # split into train and test data
    train_data, test_data = split_data_pytorch(relevant_columns)

    # create dataset and dataloader
    training_dataset = CustomBaselineDataset(train_data, target_col)
    test_dataset = CustomBaselineDataset(test_data, target_col)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def preprocess_data_version_1(df: pd.DataFrame, text_col: str, target_col: str, tokenizer):
    batch_size = 16  # 8

    ###################### encode categorical variables #####################
    encode_target_variable(df, target_col, annotation_questions, use_label_encoder=False)

    ############################################
    # tokenize the review column already here instead of in __getitem__ with the tokenize function below, so it does
    # not have to be performed for every batch while training!
    max_tokens = tokenizer.max_model_input_sizes[checkpoint]
    dataset = ds.from_pandas(df[[text_col, "review_bomb_type", "source"]])
    # don't use padding here already, instead a data collator is later used for dynamic padding
    tokenized_dataset = dataset.map(lambda data: tokenizer(data[text_col], truncation=True, max_length=max_tokens),
                                    batched=True)
    # tokenized_dataset.set_format("torch")  # convert to pytorch dataset
    X_data = tokenized_dataset.to_pandas()
    y_data = df[[target_col]]

    # TODO move split data before tokenizing to make sure nothing leaks!

    ######################## split into train and test set ########################
    # TODO save test set as a separate csv file to make sure none of the classifiers will see it
    #  while training! and to make sure it's the same for all! (which reviews should be used as the test set ?)
    train_x, test_x, train_y, test_y = split_data_scikit(X_data, y_data, stratify_on=y_data)
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    debug_stratify = True
    if debug_stratify:
        # make sure stratification works
        print(f"\nOriginal y data: {y_data[target_col].value_counts()}")
        ratio_og = y_data[target_col].value_counts(normalize=True)
        print(f"ratio_percentage: {ratio_og.round(4) * 100}")
        print("\nAfter stratifying on y data:")
        print(f"Train y: {train_y[target_col].value_counts()}")
        print(f"ratio_percentage: {train_y[target_col].value_counts(normalize=True).round(4) * 100}")
        print(f"\nTest y: {test_y[target_col].value_counts()}")
        print(f"ratio_percentage: {test_y[target_col].value_counts(normalize=True).round(4) * 100}")

    # TODO Reihenfolge der Reviews für Review Analyse wichtig! Temporal splitting (z.B. nur Reviews bis zu einem
    #  bestimmten Zeitpunkt fürs Training nutzen und alle neueren als Testdaten)
    #  -> kein Random Split oder K-Fold-Crossvalidation  (problematisch, da so im Train Set Daten (aus Sicht der
    #  Testdaten) zukünftigen Informationen verwendet werden könnten)

    ######################## create custom dataset and dataloader #######################
    # TODO also test other ways to vectorize with word embeddings such as word2vec or GloVe ?
    """
    def tokenize_review(review: str):
        max_length = tokenizer.max_model_input_sizes[checkpoint]
        encoding = tokenizer(review, return_tensors='pt', truncation=True, max_length=max_length)  # padding="longest"
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        # resize the tensors to the same size
        # input_ids = nn.functional.pad(input_ids, (0, max_length - input_ids.shape[0]), value=0)
        # attention_mask = nn.functional.pad(attention_mask, (0, max_length - attention_mask.shape[0]), value=0)
        return input_ids, attention_mask
    """

    # use a data collator to pad the tokens to the longest per batch (see "Dynamic Padding" on https://huggingface.co/learn/nlp-course/en/chapter3/2?fw=pt#dynamic-padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = CustomDataset(train_x, train_y)  # transform=tokenize_review
    test_dataset = CustomDataset(test_x, test_y)  # transform=tokenize_review
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)  # num_workers = 2
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    """
    # test if the batches have the correct shape
    for batch in train_dataloader:
        print({k: v.shape for k, v in batch.items()})
        print(tokenizer.decode(batch['input_ids'][0]))  # show the encoded version of one review
        break
    """
    return train_dataloader, test_dataloader


def train_and_predict(data: pd.DataFrame, text_col: str, tag: str):
    # create tokenizer for preprocessing the data
    # bert_tokenizer = BertTokenizer.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  # uses a Rust-based fast tokenizer version instead of Python

    should_train_model = True
    if should_train_model:
        # set all the random seeds to make everything reproducible for preprocessing and training
        classification_utils.set_random_seed()
        # random shuffle the data
        shuffled_data = data.sample(frac=1)

        # TODO also split per rb incident for training?

        train_data_loader, test_data_loader = preprocess_data_version_1(shuffled_data, text_col, target_column, tokenizer)

        # create the model as well as the training parameters and start training
        num_classes = shuffled_data[target_column].nunique()
        model = BERTClassifier(num_classes, model_checkpoint=checkpoint).to(device)
        # model = get_pretrained_bert_for_sequence(num_classes, model_checkpoint=checkpoint).to(device)   # use this for BertForSequenceClassification
        classify_review_bombing(model, train_data_loader, test_data_loader, tag)

        # load the best model for further code
        # classification_utils.load_model_checkpoint(model, optimizer, MODEL_FOLDER / f"baseline-{tag}-best-model.pt")

    should_predict = False
    if should_predict:
        # load local model
        n_classes = data[target_column].nunique()
        model = BERTClassifier(n_classes, model_checkpoint=checkpoint).to(device)
        model_checkpoint = torch.load(MODEL_FOLDER / f"baseline-{tag}-best-model_{ckp_clean}.pt")
        model.load_state_dict(model_checkpoint['model_state_dict'])
        # model.eval()  # switch to inference mode to not resume training

        # test prediction
        test_review = "The game was great and I really enjoyed the combat and the story."
        predicted_label = predict_label(test_review, target_column, model, tokenizer, device)
        print(test_review)
        print(f"Predicted label: \"{predicted_label}\"")


if __name__ == "__main__":
    print(f"Using PyTorch version: {torch.__version__}")
    utils.enable_max_pandas_display_size()

    if not MODEL_FOLDER.is_dir():
        MODEL_FOLDER.mkdir()

    device = get_pytorch_device()
    # checkpoint = "google-bert/bert-base-uncased"
    checkpoint = "distilbert-base-uncased"
    # checkpoint = "FacebookAI/roberta-base"    # there is no uncased version
    ckp_clean = re.sub("/", "-", checkpoint)  # "clean" version without the / so it can be used in filenames

    # load relevant data
    num_rows = 30    # TODO for testing
    combined_annotated_data = pd.read_csv(INPUT_DATA_FOLDER / "combined_final_annotation_all_projects_updated.csv",
                                          nrows=num_rows)
    print(f"Using {len(combined_annotated_data)} rows/reviews for training ...")

    # apply_standard_text_preprocessing(combined_annotated_data, text_col="review", remove_stopwords=False, remove_punctuation=False)
    text_column = "review"

    reviews_to_use = "both"   # "steam" / "metacritic" / "both"
    if reviews_to_use == "steam":
        print("[INFO] Training model with only steam reviews!\n")
        relevant_data = combined_annotated_data[combined_annotated_data["source"] == "Steam"].reset_index(drop=True)
        model_tag = "steam"
    elif reviews_to_use == "metacritic":
        print("[INFO] Training model with only metacritic reviews!\n")
        relevant_data = combined_annotated_data[combined_annotated_data["source"] == "Metacritic"].reset_index(drop=True)
        model_tag = "metacritic"
    elif reviews_to_use == "both":
        print("[INFO] Training model with steam and metacritic reviews!\n")
        relevant_data = combined_annotated_data
        model_tag = "both"
    else:
        raise ValueError("The specified reviews_to_use type is unknown!")

    classify_rb = True     # if False classify off_topic column
    target_column = "is-review-bombing" if classify_rb else "is-rating-game-related"

    train_and_predict(relevant_data, text_column, model_tag)
