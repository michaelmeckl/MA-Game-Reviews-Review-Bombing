#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Pyabsa needs transformers==4.29.0 and protobuf < 4.0 to work
"""

import pprint
import pathlib
import random
import pandas as pd
from nltk import tokenize
# noinspection PyPep8Naming
from pyabsa import AspectTermExtraction as ATEPC, AspectPolarityClassification as APC, \
    AspectSentimentTripletExtraction as ASTE, available_checkpoints, ModelSaveOption


def pyabsa_aspect_term_extraction():
    # you can view all available checkpoints by calling available_checkpoints()
    checkpoint_map = available_checkpoints()
    print(checkpoint_map)

    aspect_extractor = ATEPC.AspectExtractor('multilingual',  # 'english', ...
                                             auto_device=False,  # False means load model on CPU
                                             cal_perplexity=True)

    # instance inference
    input_text = 'I love this movie, it is so great!'
    atepc_result = aspect_extractor.predict([input_text],
                                            save_result=False,
                                            print_result=True,
                                            ignore_error=True,
                                            pred_sentiment=True
                                            )
    pprint.pprint(atepc_result)

    result = pd.DataFrame(
        {
            "aspect": atepc_result[0]["aspect"],
            "sentiment": atepc_result[0]["sentiment"],
            'probability': atepc_result[0]['probs'],
            "confidence": [round(x, 4) for x in atepc_result[0]["confidence"]],
            "position": atepc_result[0]["position"],
        }
    )
    print(result.head(5))


# aspect term extraction with pred_sentiment=True seems to provide more useful information than this
def pyabsa_aspect_sentiment_analysis():
    checkpoint_map = available_checkpoints(show_ckpts=True)
    print(checkpoint_map)

    # aspect polarity
    classifier = APC.SentimentClassifier('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True)

    # instance inference
    apc_result = classifier.predict(['I love this movie, it is so great!'],
                                    save_result=False,
                                    print_result=True,
                                    ignore_error=True,
                                    )
    pprint.pprint(apc_result)


def pyabsa_aspect_sentiment_triplet_extraction():
    input_text = 'I love this movie, it is so great!'
    triplet_extractor = ASTE.AspectSentimentTripletExtractor("multilingual")
    result = triplet_extractor.predict(input_text)
    pprint.pprint(result)


def configure_pyabsa():
    # example how to change default configuration and use custom-configured trainer
    config = ASTE.ASTEConfigManager.get_aste_config_multilingual()
    config.pretrained_bert = 'microsoft/deberta-v3-base'  # default: yangheng/deberta-v3-base-absa-v1.1
    config.evaluate_begin = 5
    config.max_seq_len = 80
    config.num_epoch = 30
    config.log_step = 10
    config.dropout = 0
    config.cache_dataset = False
    config.l2reg = 1e-8
    config.lsa = True
    config.seed = [random.randint(0, 10000) for _ in range(3)]

    trainer = ASTE.ASTETrainer(
        config=config,
        # from_checkpoint='english',
        checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
        auto_device=True,
    )
    triplet_extractor = trainer.load_trained_model()
    return triplet_extractor


def test_pyabsa():
    num_rows = 10
    reviews_file = pathlib.Path(
        __file__).parent.parent / "data_for_analysis" / "steam" / "steam_user_reviews_Hogwarts_Legacy_old.csv"
    df = pd.read_csv(reviews_file, nrows=num_rows)

    example_data = list(df['content'])
    print(str(len(example_data)) + " examples")
    print(example_data)

    # split into sentences before and pass every single sentence instead of whole text
    # => both work a lot better if used on individual sentences apparently

    sentences = []
    for text in example_data:
        sentence_list = tokenize.sent_tokenize(text)
        # very simple way to exclude extremely short sentences; probably should count individual tokens instead
        for sent in sentence_list:
            if len(sent) > 25:
                sentences.append(sent)
    pprint.pprint(sentences)

    # aspect extraction
    aspect_extractor = ATEPC.AspectExtractor('multilingual',  # 'english', ...
                                             auto_device=True,  # False means load model on CPU
                                             cal_perplexity=True)

    atepc_result = aspect_extractor.predict(sentences, save_result=False, ignore_error=True)
    # print(atepc_result)
    atepc_df = pd.DataFrame(atepc_result)
    atepc_df.to_csv("./pyabsa_atepc_result.csv", index=False)

    # TODO does this work?
    triplet_extractor = ASTE.AspectSentimentTripletExtractor("multilingual")
    aste_result = triplet_extractor.predict(sentences, ignore_error=True)
    # pprint.pprint(aste_result)
    aste_df = pd.DataFrame(aste_result)
    aste_df.to_csv("./pyabsa_aste_result.csv", index=False)


if __name__ == "__main__":
    # pyabsa_aspect_term_extraction()
    # pyabsa_aspect_sentiment_analysis()
    # pyabsa_aspect_sentiment_triplet_extraction()
    test_pyabsa()
