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
from classification.classification_constants import (MODEL_FOLDER, INPUT_DATA_FOLDER, annotation_questions, RANDOM_SEED,
                                                     TRAIN_TEST_DATA_FOLDER, PLOT_FOLDER)
from classification.classification_utils import split_data_scikit, encode_target_variable, split_data_pytorch, \
    get_pytorch_device, show_class_distributions, create_test_train_set
from classification.bert_classifier import BERTClassifier, predict_label, evaluate_model, train_model, \
    predict_test_labels
from classification.custom_datasets import CustomBaselineDataset, CustomDataset
from sentiment_analysis_and_nlp.nlp_utils import apply_standard_text_preprocessing
from utils import utils
import optuna

best_results = {'loss': float('inf'), 'model': None}


def classify_review_bombing(trial: optuna.Trial, hyperparams, bert_model, train_dataloader: DataLoader,
                            test_dataloader: DataLoader, tag: str, num_epochs=5):
    num_epochs = hyperparams['epoch']
    print("Using num_epochs: ", num_epochs)

    total_steps = len(train_dataloader) * num_epochs
    # use nn.BCEWithLogitsLoss() instead? -> switch num_classes to 1 and adjust loss calculation in train/eval loop
    loss_function = nn.CrossEntropyLoss()
    learning_rate = hyperparams['learning_rate']
    print("Using learning_rate: ", learning_rate)
    optimizer = torch.optim.AdamW(bert_model.parameters(),
                                  lr=learning_rate)  # 3e-5, 5e-5  # see BERT paper for learning rates
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
        val_loss, val_accuracy, report = evaluate_model(bert_model, test_dataloader, loss_function, device, epoch,
                                                        writer,
                                                        train_history)

        classification_utils.save_model_checkpoint(bert_model, optimizer, epoch,
                                                   output_path=MODEL_FOLDER / f"baseline-{tag}-epoch-{epoch}_{ckp_clean}.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Best val loss is now: {val_loss:.2f} (val accuracy: {val_accuracy:.2f}%) \n")
            classification_utils.save_model_checkpoint(bert_model, optimizer, epoch,
                                                       output_path=MODEL_FOLDER / f"baseline-{tag}-best-model_{ckp_clean}.pt")
            # save over all trials:
            if best_loss < best_results['loss']:
                print("NEW BEST MODEL FOUND. Validation loss: ", best_loss)
                best_results['loss'] = best_loss
                best_results['model'] = bert_model.state_dict()

        trial.report(val_loss, epoch)

        if epoch == (num_epochs - 1):
            # save the last classification report to file
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(MODEL_FOLDER / f"baseline-report-epoch_{epoch}-{tag}-{ckp_clean}.csv")

    print("Finished training the model!\n")
    finish_time = time.time()
    total_time = finish_time - start_time
    time_per_epoch = total_time / num_epochs
    print(f"Time for training overall: {total_time:.2f} seconds / {(total_time / 60):.2f} min")
    print(f"Time per epoch: {time_per_epoch:.2f} seconds / {(time_per_epoch / 60):.2f} min")

    writer.flush()
    writer.close()
    # plot the training history (loss and accuracy)
    # classification_utils.show_training_plot(train_history["train_accuracy"], train_history["val_accuracy"],
    #                                        train_history["train_loss"], train_history["val_loss"], train_history["f1_score"],
    #                                        output_name=f"train_history_{tag}_{ckp_clean}", show=False)

    return best_loss


"""
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

    # split into train and val data
    train_data, val_data = split_data_pytorch(relevant_columns)

    # create dataset and dataloader
    training_dataset = CustomBaselineDataset(train_data, text_col, target_col)
    val_dataset = CustomBaselineDataset(val_data, text_col, target_col)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
"""


def preprocess_data(df: pd.DataFrame, text_col: str, target_col: str, tokenizer, batch_size=16):
    ######################## split into train and validation set ########################
    # X_data = df[[text_col, "review_bombing_incident", "review_bomb_type", "source"]]
    X_data = df.loc[:, df.columns != target_col]  # select everything but target col
    y_data = df[[target_col]]

    # unfortunately it seems only one additional column can be specified here! "review_bombing_incident" seems to be
    # the best one for the resulting strata
    train_x, val_x, train_y, val_y = split_data_scikit(X_data, y_data, stratify_on=df[
        [target_column, "review_bombing_incident"]])  # "source"
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    val_x = val_x.reset_index(drop=True)
    val_y = val_y.reset_index(drop=True)

    debug_stratify = False
    if debug_stratify:
        show_class_distributions(X_data, y_data, train_x, train_y, val_x, val_y, target_col)

    ############################################
    # tokenize the review column already here instead of in __getitem__ with the tokenize function below, so it does
    # not have to be performed for every batch while training!
    max_tokens = tokenizer.max_model_input_sizes[checkpoint]
    train_dataset = ds.from_pandas(train_x)
    val_dataset = ds.from_pandas(val_x)
    # don't use padding here already, instead a data collator is later used for dynamic padding
    tokenized_train_dataset = train_dataset.map(
        lambda data: tokenizer(data[text_col], truncation=True, max_length=max_tokens),
        batched=True)
    tokenized_val_dataset = val_dataset.map(
        lambda data: tokenizer(data[text_col], truncation=True, max_length=max_tokens),
        batched=True)
    # tokenized_train_dataset.set_format("torch")  # convert to pytorch dataset
    train_x_tokenized = tokenized_train_dataset.to_pandas()
    val_x_tokenized = tokenized_val_dataset.to_pandas()

    ######################## create custom dataset and dataloader #######################
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

    train_dataset = CustomDataset(train_x_tokenized, train_y)  # transform=tokenize_review
    val_dataset = CustomDataset(val_x_tokenized, val_y)  # transform=tokenize_review
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=data_collator)  # num_workers = 2
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    """
    # test if the batches have the correct shape
    for batch in train_dataloader:
        print({k: v.shape for k, v in batch.items()})
        print(tokenizer.decode(batch['input_ids'][0]))  # show the encoded version of one review
        break
    """
    return train_dataloader, val_dataloader


def train_baseline_model(trial: optuna.Trial):
    hyperparams = suggest_hyperparameters(trial)
    batch_size = hyperparams['batch_size']
    print("Using batch size: ", batch_size)

    # random shuffle the data (without it there could be some biases and it would be easier to overfit)
    shuffled_set = train_set.sample(frac=1, random_state=RANDOM_SEED)

    if use_subset:
        shuffled_set = shuffled_set.head(100)
    print(f"[INFO] Using {len(shuffled_set)} rows/reviews for training ...")

    train_data_loader, val_data_loader = preprocess_data(shuffled_set, text_column, target_column,
                                                         pre_trained_tokenizer, batch_size)

    # create the model as well as the training parameters and start training
    num_classes = shuffled_set[target_column].nunique()
    model = BERTClassifier(num_classes, model_checkpoint=checkpoint).to(device)
    # model = get_pretrained_bert_for_sequence(num_classes, model_checkpoint=checkpoint).to(device)   # use this for BertForSequenceClassification
    best_loss = classify_review_bombing(trial, hyperparams, model, train_data_loader, val_data_loader, model_tag)

    return best_loss


def suggest_hyperparameters(trial: optuna.Trial):
    return {
        'learning_rate': trial.suggest_categorical('learning_rate', [2e-5, 3e-5, 5e-5]),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16]),
        'epoch': trial.suggest_categorical('epoch', [3, 4, 5]),
    }


def optuna_optim():
    n_trials = 18
    # TODO use GridSampler to prevent duplicate combinations; always adjust dict to values in function above
    sampler = optuna.samplers.GridSampler({
        'learning_rate': [2e-5, 3e-5, 5e-5],
        'batch_size': [8, 16],
        'epoch': [3, 4, 5],
    })
    study: optuna.Study = optuna.create_study(study_name="baseline-hyperparameter-optimization", sampler=sampler)
    study.optimize(train_baseline_model, n_trials=n_trials)
    # train_baseline_model(train_set, pre_trained_tokenizer, text_column, model_tag)

    print("\n#####################################################\n")
    # Gives the best loss value
    print(study.best_value)

    # Gives the best hyperparameter values to get the best loss value
    print(study.best_params)

    # Return info about best Trial such as start and end datetime, hyperparameters
    print(study.best_trial)


def predict_on_test_data(test_data: pd.DataFrame, tokenizer, text_col: str, tag: str, batch_size=16, incident_positive=False):
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
    print('Predicting labels for {:,} test reviews ...'.format(len(test_dataset)))

    n_classes = test_data[target_column].nunique()
    model = BERTClassifier(n_classes, model_checkpoint=checkpoint).to(device)
    model_checkpoint = torch.load(MODEL_FOLDER / f"baseline-{tag}-best-model_{ckp_clean}.pt")
    model.load_state_dict(model_checkpoint['model_state_dict'])

    predicted_labels = predict_test_labels(model, test_dataloader, device, incident_positive)
    # show some predictions
    prediction_results = test_data[[text_col, target_column]]
    prediction_results.insert(2, "predictions", predicted_labels)

    # convert 0 and 1 back to categorical label
    encoding = {
        0: "Is Review Bombing" if target_column == "is-review-bombing" else "Is Off-Topic",
        1: "Not Review Bombing" if target_column == "is-review-bombing" else "Not Off-Topic"
    }
    prediction_results = prediction_results.replace(encoding)

    for idx, row in prediction_results.iloc[:10].iterrows():
        print(f'Review: {row[text_col]}\nActual label: {row[target_column]}\nPredicted label: {row["predictions"]}\n')


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

    device = get_pytorch_device()
    checkpoint = "google-bert/bert-base-uncased"
    # checkpoint = "distilbert-base-uncased"
    # checkpoint = "FacebookAI/roberta-base"    # there is no uncased version
    ckp_clean = re.sub("/", "-", checkpoint)  # "clean" version without the / so it can be used in filenames

    # bert_tokenizer = BertTokenizer.from_pretrained(checkpoint)
    # uses a Rust-based fast tokenizer version instead of Python
    pre_trained_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    create_new_test_train_data = False
    if create_new_test_train_data:
        train_set, test_set = create_test_train_set()
    else:
        # load existing data
        train_set = pd.read_csv(TRAIN_TEST_DATA_FOLDER / "train_data.csv")
        test_set = pd.read_csv(TRAIN_TEST_DATA_FOLDER / "test_data.csv")
        print(f"Using {len(train_set)} reviews as train set.")
        print(f"Using {len(test_set)} reviews as test set.")

    reviews_to_use = "both"  # "steam" / "metacritic" / "both"
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

    # update the tag for the model save
    model_tag = f"{model_tag}_review_bombing" if classify_rb else f"{model_tag}_game_related"

    # set all the random seeds to make everything reproducible for preprocessing and training
    classification_utils.set_random_seed()

    optuna_optim()

    """
    # predict on positive and on negative test incident
    test_set_positive = test_set[test_set["review_bomb_type"] == "positiv"]
    test_set_negative = test_set[test_set["review_bomb_type"] == "negativ"]
    test_set_positive = test_set_positive.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_set_negative = test_set_negative.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    predict_on_test_data(test_set_positive, pre_trained_tokenizer, text_column, model_tag, incident_positive=True)
    predict_on_test_data(test_set_negative, pre_trained_tokenizer, text_column, model_tag)
    """
