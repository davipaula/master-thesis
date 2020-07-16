"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from modeling.smash_dataset import SMASHDataset
from modeling.smash_rnn_model import SmashRNNModel
import csv
import numpy as np
import argparse
from src.utils.utils import (
    get_words_per_document_at_word_level,
    get_document_at_word_level,
    get_document_at_sentence_level,
    get_words_per_sentence_at_sentence_level,
    get_sentences_per_paragraph_at_sentence_level,
)

PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"
TEXT_IDS_COLUMN = "text_ids"
CLICK_RATE_COLUMN = "click_rate"
TARGET_ARTICLE_COLUMN = "target_article"
SOURCE_ARTICLE_COLUMN = "source_article"

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

WIKI_ARTICLES_DATASET_PATH = "./data/dataset/wiki_articles_english_complete.csv"

MODEL_FOLDER = "./trained_models/"
FULL_DATASET_PATH = "./data/dataset/click_stream_train.pth"
WORD2VEC_PATH = "./data/source/glove.6B.50d.txt"
TEST_DATASET_PATH = "./data/dataset/click_stream_test.pth"
RESULTS_PATH = "./results/"


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        device = torch.device("cuda")
    else:
        torch.manual_seed(123)
        device = torch.device("cpu")

    logger.info("Initializing parameters")

    click_stream_test = torch.load(TEST_DATASET_PATH)

    batch_size = 6
    test_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
    }
    test_generator = torch.utils.data.DataLoader(click_stream_test, **test_params)

    criterion = nn.MSELoss().to(device)

    model = load_model(MODEL_FOLDER, opt.level, WORD2VEC_PATH)
    model.to(device)

    articles = SMASHDataset(WIKI_ARTICLES_DATASET_PATH)

    loss_list = []
    columns_names = [
        "model",
        "source_article",
        "target_article",
        "actual_click_rate",
        "predicted_click_rate",
    ]
    predictions_list = pd.DataFrame(columns=columns_names)

    logger.info(f"Model Smash-RNN {opt.level} level. Starting evaluation")

    for row in tqdm(test_generator):
        source_articles = articles.get_articles(row[SOURCE_ARTICLE_COLUMN])
        target_articles = articles.get_articles(row[TARGET_ARTICLE_COLUMN])

        if opt.level == "sentence":
            source_articles = transform_to_sentence_level(source_articles)
            target_articles = transform_to_sentence_level(target_articles)

        elif opt.level == "word":
            source_articles = transform_to_word_level(source_articles)
            target_articles = transform_to_word_level(target_articles)

        row[CLICK_RATE_COLUMN] = row[CLICK_RATE_COLUMN].to(device)
        source_articles[TEXT_IDS_COLUMN] = source_articles[TEXT_IDS_COLUMN].to(device)
        source_articles[WORDS_PER_SENTENCE_COLUMN] = source_articles[WORDS_PER_SENTENCE_COLUMN].to(device)
        source_articles[SENTENCES_PER_PARAGRAPH_COLUMN] = source_articles[SENTENCES_PER_PARAGRAPH_COLUMN].to(device)
        source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN] = source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN].to(device)
        target_articles[TEXT_IDS_COLUMN] = target_articles[TEXT_IDS_COLUMN].to(device)
        target_articles[WORDS_PER_SENTENCE_COLUMN] = target_articles[WORDS_PER_SENTENCE_COLUMN].to(device)
        target_articles[SENTENCES_PER_PARAGRAPH_COLUMN] = target_articles[SENTENCES_PER_PARAGRAPH_COLUMN].to(device)
        target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN] = target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN].to(device)

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
                [opt.level] * 32,
                row[SOURCE_ARTICLE_COLUMN],
                row[TARGET_ARTICLE_COLUMN],
                row[CLICK_RATE_COLUMN].tolist(),
                predictions.squeeze(1).tolist(),
            ),
            columns=columns_names,
        )

        predictions_list = predictions_list.append(batch_results, ignore_index=True)

    final_loss = sum(loss_list) / len(loss_list)

    predictions_list.to_csv(f"./results/test/results_{opt.level}_level.csv", index=False)

    logger.info(f"Model Smash-RNN {opt.level} level. Evaluation finished. Final loss: {final_loss}")


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
    parser.add_argument("--level", type=str, default="paragraph")
    parser.add_argument("--batch_size", type=int, default=2)

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
    document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_sentence_at_sentence_level(document[WORDS_PER_SENTENCE_COLUMN])
    document[SENTENCES_PER_PARAGRAPH_COLUMN] = get_sentences_per_paragraph_at_sentence_level(
        document[SENTENCES_PER_PARAGRAPH_COLUMN]
    )
    document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(batch_size, dtype=int)

    return document


if __name__ == "__main__":
    opt = get_args()
    test(opt)
