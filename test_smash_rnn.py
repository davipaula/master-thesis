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

from utils.constants import RESULT_FILE_COLUMNS_NAMES

from utils.utils import get_word2vec_path, get_model_name

PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"
TEXT_IDS_COLUMN = "text_ids"
CLICK_RATE_COLUMN = "click_rate"
TARGET_ARTICLE_COLUMN = "target_article"
SOURCE_ARTICLE_COLUMN = "source_article"

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

WIKI_ARTICLES_DATASET_PATH = "./data/dataset/wiki_articles_english_complete.csv"

MODEL_FOLDER = "./trained_models/"
FULL_DATASET_PATH = "./data/dataset/click_stream_train.pth"
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

    batch_size = opt.batch_size
    test_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
    }
    test_generator = torch.utils.data.DataLoader(click_stream_test, **test_params)

    criterion = nn.MSELoss().to(device)

    model_name = get_model_name(opt.level, opt.model_name, opt.introduction_only)

    model = load_model(MODEL_FOLDER, model_name, opt)
    model.to(device)

    articles = SMASHDataset(
        WIKI_ARTICLES_DATASET_PATH, introduction_only=opt.introduction_only
    )

    loss_list = []
    predictions_list = pd.DataFrame(columns=RESULT_FILE_COLUMNS_NAMES)

    logger.info(f"Model Smash-RNN {opt.level} level. Starting evaluation")

    i = 0
    for row in tqdm(test_generator):
        source_articles = articles.get_articles(row[SOURCE_ARTICLE_COLUMN])
        target_articles = articles.get_articles(row[TARGET_ARTICLE_COLUMN])

        row[CLICK_RATE_COLUMN] = row[CLICK_RATE_COLUMN].to(device)

        predictions = model(target_articles, source_articles)

        loss = criterion(predictions.squeeze(1), row[CLICK_RATE_COLUMN])
        loss_list.append(loss)

        batch_results = pd.DataFrame(
            zip(
                [model_name] * batch_size,
                row[SOURCE_ARTICLE_COLUMN],
                row[TARGET_ARTICLE_COLUMN],
                row[CLICK_RATE_COLUMN].tolist(),
                predictions.squeeze(1).tolist(),
            ),
            columns=RESULT_FILE_COLUMNS_NAMES,
        )

        predictions_list = predictions_list.append(batch_results, ignore_index=True)

    final_loss = sum(loss_list) / len(loss_list)

    predictions_list.to_csv(
        f"./results/test/results_{opt.level}_level_{model_name}.csv", index=False
    )

    logger.info(
        f"Model Smash-RNN {opt.level} level. Evaluation finished. Final loss: {final_loss}"
    )


def load_model(model_folder, model_name, opt):
    # Load from txt file (in word2vec format)
    word2vec_path = get_word2vec_path(opt.w2v_dimension)
    dict = pd.read_csv(
        filepath_or_buffer=word2vec_path,
        header=None,
        sep="\s",
        engine="python",
        quoting=csv.QUOTE_NONE,
    ).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(
        np.concatenate([unknown_word, dict], axis=0).astype(np.float)
    )

    # Siamese + Attention model
    model = SmashRNNModel(dict, dict_len, embed_dim, opt.level)

    model_path = f"{model_folder}{model_name}_model.pt"
    logger.info(f"Model path: {model_path}")
    if torch.cuda.is_available():
        model_state_dict = torch.load(model_path)
    else:
        model_state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage
        )

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
    parser.add_argument("--paragraphs_limit", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="base")
    parser.add_argument("--w2v_dimension", type=int, default=50)
    parser.add_argument("--introduction_only", type=bool, default=False)

    return parser.parse_args()


def transform_to_word_level(document, device):
    batch_size = document[TEXT_IDS_COLUMN].shape[0]

    document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_document_at_word_level(
        document[WORDS_PER_SENTENCE_COLUMN]
    )
    document[TEXT_IDS_COLUMN] = get_document_at_word_level(
        document[TEXT_IDS_COLUMN], document[WORDS_PER_SENTENCE_COLUMN], device
    )
    document[SENTENCES_PER_PARAGRAPH_COLUMN] = torch.ones((batch_size, 1), dtype=int)
    document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(batch_size, dtype=int)

    return document


def transform_to_sentence_level(document, device):
    batch_size = document[TEXT_IDS_COLUMN].shape[0]

    document[TEXT_IDS_COLUMN] = get_document_at_sentence_level(
        document[TEXT_IDS_COLUMN], device
    )
    document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_sentence_at_sentence_level(
        document[WORDS_PER_SENTENCE_COLUMN], device
    )
    document[
        SENTENCES_PER_PARAGRAPH_COLUMN
    ] = get_sentences_per_paragraph_at_sentence_level(
        document[SENTENCES_PER_PARAGRAPH_COLUMN]
    )
    document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(batch_size, dtype=int)

    return document


if __name__ == "__main__":
    _opt = get_args()
    test(_opt)
