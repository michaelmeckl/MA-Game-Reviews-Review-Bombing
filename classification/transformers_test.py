#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
import numpy as np
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding, \
    create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from datasets import Dataset, load_dataset
import evaluate
import pandas as pd
import tensorflow as tf
import torch
import pprint
from sentiment_analysis_and_nlp.nlp_utils import detect_language
from useful_code_from_other_projects import utils


# TODO only for testing:
example_reviews_hogwarts_legacy = [
    "Hairy Pot",
    "Denuvo = 0, you pay to play a worse version of the game and that's a 0 dc the game",
    "Disgusted the see that the entire story of this game is just antisemitic myths and conspiracies with the names switched out.",
    "The stupid propaganda ruined this game , it would be nice if they stop forcing those gay characters in every game , studios are scared to get canceled...",
    "Brilliant. Impressive attention to details. Immersive open world, so many things to do around. So many things to discover. We are so lucky to have it during out timelines",
    "I cannot believe that the creators of Hogwarts Legacy and JK Rowling would do something so controversial. It is entirely unacceptable in 2023 to have these view and they deserve all the hate they get. I will be refunding this game and boycotting it to spread awareness. It is inhumane and downright demeaning that they make you play as british person"
]

example_reviews_cyberpunk = [
    "My advice…… never play cyberpunk….. search up the games issues on the internet and find out y’all’s self",
    "Shallow piece of garbage",
    "Wo bist du, wenn die USA und NATO andere Länder eindringen",
    "Verbrauchertäuschung und Doppelmoral in der Politik, widerlich.",
    "play this if your brain is made of slime and you look ugly af",
    "uhhh i dunno i just got here",
    "it's worst game ever not just because of bugs. but gamplay mechanic, AI, npc,s all are worst.revewers got paid for sure",
    "I am sorry CDPR but single player and RPG games in general are alive and well. Dont even think about moving in to online services games for easy cash grabs while proving garbage games to your followers"
]

example_posts = ["Review bombed? The game is not working properly? What are they thinking???",
                 "You see how quick they are to make videos and articles whenever fanboys review bomb a game. Yet are silent when an outlet like Wire essentially does the same by giving Hogwarts Legacy a 1/10.",
                 "I'm just shocked game journos didn't review bomb Hogwarts Legacy",
                 "Remember when these guys tried to 'review bomb' Hogwarts Legacy? How did that turn out?",
                 "Hogwart's Legacy is a special case on account of the huge backlash the game is getting simply because it is part of J.K. Rowling's franchise",
                 "Star Wars haters will probably review bomb Survivor cause they're toxic, Sony haters will review bomb Spider-Man 2",
                 "Review bombing is disgusting. That being said, this isn't a review bombing, the game is literally almost unplayable",
                 "**People are now Review Bombing Cyberpunk cause it won Labor of Love",
                 "Why is the game being review bombed once again?",
                 "Trolls Review Bomb Cyberpunk 2077 After Devs Speak Out Against Russia’s Invasion Of Ukraine"]


def test_tokenization(input_text: list[str] = None, checkpoint="distilbert-base-uncased-finetuned-sst-2-english"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = input_text if input_text is not None else [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
    # print(inputs)
    return inputs


def train_tensorflow_model(inputs, checkpoint="distilbert-base-uncased-finetuned-sst-2-english"):
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    predictions = tf.math.softmax(outputs.logits, axis=-1)
    return predictions


""" 
def train_pytorch_model(inputs, checkpoint="distilbert-base-uncased-finetuned-sst-2-english"):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions
"""


def test_named_entity_recognition():
    classifier = pipeline(task="ner")  # , grouped_entities=True
    preds = classifier("Hugging Face is a French company based in New York City.")
    preds = [
        {
            "entity": pred["entity"],
            "score": round(pred["score"], 4),
            "index": pred["index"],
            "word": pred["word"],
            "start": pred["start"],
            "end": pred["end"],
        }
        for pred in preds
    ]
    print(*preds, sep="\n")


def test_sentiment_analysis(df: pd.DataFrame, model_checkpoint="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    By default, the HuggingFace sentiment analysis pipeline uses the DistillBert model and was fine-tuned with the
    Stanford Sentiment Treebank v2 (SST2) dataset.
    Limitation: Neutral category is missing
    """
    sentiment_classifier = pipeline(task='sentiment-analysis', model=model_checkpoint)

    # Passing the tweets into the sentiment pipeline and extracting the sentiment score and label
    df = (
        df
        .assign(sentiment=lambda x: x['content'].apply(lambda s: sentiment_classifier(s)))
        .assign(
            label=lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
            score=lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
        )
    )
    print(df.head())


def load_example_data():
    reviews_file = pathlib.Path(
        __file__).parent / "data_for_analysis" / "steam" / "steam_user_reviews_Hogwarts_Legacy_old.csv"
    df = pd.read_csv(reviews_file)
    # df = utils.remove_linebreaks_from_pd_cells(df, column_name="content")
    # df = df[df["content"].apply(lambda x: detect_language(x)).eq('en', 'de')]

    example_data = list(df['content'].head(10))
    example_data.extend(example_reviews_hogwarts_legacy)

    print("\n#####################################")
    print(str(len(example_data)) + " examples")
    # pprint.pprint(example_data)
    print("\n#####################################")

    # tokenize input text and train AutoModel Classifier
    # TODO Test other Encoder Models from HuggingFace such as Albert, Roberta oder DistilBert ?
    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenized_inputs = test_tokenization(example_data, model_ckpt)
    predictions = train_tensorflow_model(tokenized_inputs, model_ckpt)

    correct_predictions = 0
    print("Prediction Results (TFAutoModelForSequenceClassification):")
    for prediction, input_sentence in zip(predictions, example_data):
        prediction_idx = np.argmax(prediction)
        if prediction_idx == 0:
            result_prediction = "Negative"
            correct_predictions += 1  # all input examples were negative
        else:
            result_prediction = "Positive"

        # manchmal werden leider Sätze, die eigentlich offensichtlich negativ sind, als positiv eingeschätzt :(
        print(f"{result_prediction} ({prediction[prediction_idx] * 100:.2f} %) for text: \"{input_sentence}\"")

    print(f"\nOverall {correct_predictions}/{len(predictions)} ({correct_predictions / len(predictions):.2f}%) correct")
    print("\n#####################################")

    # Die sentiment-analyse-pipeline liefert die exakt gleichen Ergebnisse wie TFAutoModelForSequenceClassification
    # oberhalb (vermutlich weil gleiches Model verwendet lol ...)
    """
    sentiment_classifier = pipeline(task='sentiment-analysis', model=model_ckpt)
    preds = sentiment_classifier(example_data)
    preds = [{"text": data, "score": round(pred["score"], 4), "label": pred["label"]} for pred, data in zip(preds, example_data)]
    print("Prediction Results (sentiment-analysis):")
    for prediction in preds:
        print(f"{prediction['label']} ({prediction['score'] * 100:.2f} %) for text: \"{prediction['text']}\"")
    """


def convert_to_huggingface_dataset(data):
    # Option 1 - csv
    dataset = load_dataset("csv", data_files="my_file.csv")
    # Option 2 - pandas df
    # dataset = Dataset.from_pandas(df)
    # Option 3 - dict
    # dataset = Dataset.from_dict(dictionary)


def test_imdb_tutorial():
    num_rows = 100
    reviews_file = pathlib.Path(
        __file__).parent.parent / "data_for_analysis" / "steam" / "steam_user_reviews_Hogwarts_Legacy_old.csv"
    df = pd.read_csv(reviews_file, nrows=num_rows)

    # add new label column, not used at the moment
    df["label"] = 0
    df.loc[df["rating_positive"], "label"] = 1

    # result is {"False": [...]}
    # result = df.groupby('rating_positive')['content'].apply(list).to_dict()

    train_data_len = int(num_rows * 0.8)  # 80 - 20 - split

    example_data = list(df['content'][:train_data_len])
    example_data.extend(example_reviews_hogwarts_legacy)
    print(str(len(example_data)) + " examples")

    # create example train data
    # labeled_examples = [(False, el) for el in example_data]
    labeled_train_dict = {"content": example_data, "label": [0] * len(example_data)}
    labeled_train_dict["content"].append("great game, really astonishing!")  # add one positive example
    labeled_train_dict["label"].append(1)

    # create validation dataset
    valid_data = list(df['content'][train_data_len:])
    labeled_valid_dict = {"content": valid_data, "label": [0] * len(valid_data)}

    print(example_data)
    print(valid_data)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(data):
        return tokenizer(data["content"], truncation=True)

    example_dataset_train = Dataset.from_dict(labeled_train_dict)
    tokenized_dataset_train = example_dataset_train.map(preprocess_function, batched=True)
    example_dataset_valid = Dataset.from_dict(labeled_valid_dict)
    tokenized_dataset_valid = example_dataset_valid.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy_score = accuracy.compute(predictions=predictions, references=labels)['accuracy']
        f1_score = f1.compute(predictions=predictions, references=labels)['f1']
        return {"accuracy": accuracy_score, "f1": f1_score}

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_dataset_train) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    print("total train steps: ", total_train_steps)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    tf_train_set = model.prepare_tf_dataset(
        tokenized_dataset_train,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_dataset_valid,
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)  # No loss argument needed as it is automatically inferred !

    callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callback)

    # save the trained model locally (so it can be loaded later)
    local_model_path = 'model_test/'
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)

    # Test inference
    test_data = example_reviews_cyberpunk  # correctly classifies all examples as Negative

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    prediction = classifier(test_data)
    print("Predictions:\n", prediction)

    # alternative: manual inference without pipeline
    """
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    inputs = tokenizer(test_data, padding=True, truncation=True, return_tensors="tf")
    model = TFAutoModelForSequenceClassification.from_pretrained(local_model_path)
    logits = model(**inputs).logits
    
    # predicted_class_ids = tf.math.argmax(logits, axis=-1)
    predicted_class_ids = np.argmax(logits, axis=1)
    print(predicted_class_ids)
    classifications = [model.config.id2label[output] for output in predicted_class_ids]
    print(classifications)
    """


def test_huggingface_absa():
    # TODO
    pass


if __name__ == "__main__":
    # load_example_data()
    # test_imdb_tutorial()
    test_huggingface_absa()
