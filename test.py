"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import os
import pandas as pd
from comet_ml import Experiment
import torch
from torch import nn

from smash_dataset import SMASHDataset
from src.smash_rnn_model import SmashRNNModel
import csv
import numpy as np
import argparse
from src.utils import (
    get_max_lengths,
    get_words_per_document_at_word_level,
    get_document_at_word_level,
    get_document_at_sentence_level,
    get_words_per_sentence_at_sentence_level,
    get_sentences_per_paragraph_at_sentence_level,
)
from datetime import datetime

PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"
TEXT_IDS_COLUMN = "text_ids"
CLICK_RATE_COLUMN = "click_rate"
TARGET_ARTICLE_COLUMN = "target_article"
SOURCE_ARTICLE_COLUMN = "source_article"

def test(opt):
    test_generator = torch.load(opt.test_dataset_path)

    if torch.cuda.is_available():
        experiment = Experiment(api_key="NPD7aHoJxhZgG0MNWBkFb3hzZ", project_name="thesis-davi", workspace="davipaula")

    criterion = nn.MSELoss()

    model = load_model(opt.model_folder, opt.level, opt.word2vec_path)

    articles = SMASHDataset()

    loss_list = []
    columns_names = ["previous_document", "current_document", "actual_click_rate", "predicted_click_rate"]
    predictions_list = pd.DataFrame(columns=columns_names)

    print("Starting test")

    i = 0
    for row in test_generator:
        source_articles = articles.get_articles(row[SOURCE_ARTICLE_COLUMN])
        target_articles = articles.get_articles(row[TARGET_ARTICLE_COLUMN])

        if torch.cuda.is_available():
            row[CLICK_RATE_COLUMN] = row[CLICK_RATE_COLUMN].cuda()
            source_articles[TEXT_IDS_COLUMN] = source_articles[TEXT_IDS_COLUMN].cuda()
            source_articles[WORDS_PER_SENTENCE_COLUMN] = source_articles[WORDS_PER_SENTENCE_COLUMN].cuda()
            source_articles[SENTENCES_PER_PARAGRAPH_COLUMN] = source_articles[SENTENCES_PER_PARAGRAPH_COLUMN].cuda()
            source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN] = source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN].cuda()
            target_articles[TEXT_IDS_COLUMN] = source_articles[TEXT_IDS_COLUMN].cuda()
            target_articles[WORDS_PER_SENTENCE_COLUMN] = source_articles[WORDS_PER_SENTENCE_COLUMN].cuda()
            target_articles[SENTENCES_PER_PARAGRAPH_COLUMN] = source_articles[SENTENCES_PER_PARAGRAPH_COLUMN].cuda()
            target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN] = source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN].cuda()

        if opt.level == "sentence":
            source_articles = transform_to_sentence_level(source_articles)
            target_articles = transform_to_sentence_level(target_articles)

        elif opt.level == "word":
            source_articles = transform_to_word_level(source_articles)
            target_articles = transform_to_word_level(target_articles)

        predictions = model(
            target_articles[TEXT_IDS_COLUMN],
            target_articles[WORDS_PER_SENTENCE_COLUMN],
            target_articles[SENTENCES_PER_PARAGRAPH_COLUMN],
            target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN],
            source_articles[TEXT_IDS_COLUMN],
            source_articles[WORDS_PER_SENTENCE_COLUMN],
            source_articles[SENTENCES_PER_PARAGRAPH_COLUMN],
            source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN],
        )

        loss = criterion(predictions.squeeze(1), row[CLICK_RATE_COLUMN])
        loss_list.append(loss)

        batch_results = pd.DataFrame(
            zip(
                row[SOURCE_ARTICLE_COLUMN],
                row[TARGET_ARTICLE_COLUMN],
                row[CLICK_RATE_COLUMN].tolist(),
                predictions.squeeze(1).tolist(),
            ),
            columns=columns_names,
        )

        predictions_list = predictions_list.append(batch_results, ignore_index=True)

        if i == 4:
            break

        i += 1

    loss = sum(loss_list) / len(loss_list)

    if torch.cuda.is_available():
        experiment.log_metric("test_{}_level_loss".format(opt.level), loss.item())

    print("Test: loss: {}\n\n".format(loss))
    predictions_list.to_csv(
        "./results/results_{}_level_{}.csv".format(opt.level, datetime.now()), index=False
    )


def load_model(model_folder, level, word2vec_path):
    # Load from txt file (in word2vec format)
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

    # Siamese + Attention model
    model = SmashRNNModel(dict, dict_len, embed_dim)

    model_path = model_folder + os.sep + level + "_level_model.pt"
    if torch.cuda.is_available():
        model_state_dict = torch.load(model_path)
    else:
        model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(model_state_dict)
    for parameter in model.parameters():
        parameter.requires_grad = False

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    return model


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Semantic Text Matching for Long-Form Documents to predict the number of clicks for Wikipedia articles"""
    )
    parser.add_argument("--model_folder", type=str, default="./trained_models/")
    parser.add_argument("--level", type=str, default="paragraph")
    parser.add_argument("--test_dataset_path", type=str, default="./data/dataset/click_stream_test.pth")
    parser.add_argument("--word2vec_path", type=str, default="./data/glove.6B.50d.txt")

    return parser.parse_args()


def transform_to_word_level(document):
    batch_size = document[TEXT_IDS_COLUMN].shape[0]

    document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_document_at_word_level(document[WORDS_PER_SENTENCE_COLUMN])
    document[TEXT_IDS_COLUMN] = get_document_at_word_level(
        document[TEXT_IDS_COLUMN], document[WORDS_PER_SENTENCE_COLUMN]
    )
    document[SENTENCES_PER_PARAGRAPH_COLUMN] = torch.ones((batch_size, 1), dtype=int)
    document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(batch_size, dtype=int)

    return document


def transform_to_sentence_level(document):
    batch_size = document[TEXT_IDS_COLUMN].shape[0]

    document[TEXT_IDS_COLUMN] = get_document_at_sentence_level(document[TEXT_IDS_COLUMN])
    document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_sentence_at_sentence_level(
        document[WORDS_PER_SENTENCE_COLUMN]
    )
    document[SENTENCES_PER_PARAGRAPH_COLUMN] = get_sentences_per_paragraph_at_sentence_level(
        document[SENTENCES_PER_PARAGRAPH_COLUMN]
    )
    document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(batch_size, dtype=int)

    return document


if __name__ == "__main__":
    opt = get_args()
    test(opt)
