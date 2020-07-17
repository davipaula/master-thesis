import sys
import os

from data_structure.click_stream_dataset import ClickStreamDataset
from data_structure.click_stream_pre_processed import ClickStreamPreProcessed

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging
import torch
import pandas as pd

os.chdir("/Users/dnascimentodepau/Documents/python/thesis/thesis-davi")
TRAIN_DATASET_PATH = "./data/processed/train.csv"
VALIDATION_DATASET_PATH = "./data/processed/validation.csv"
TEST_DATASET_PATH = "./data/processed/test.csv"

TRAIN_FINAL_DATASET_PATH = "./data/dataset/click_stream_train.pth"
VALIDATION_FINAL_DATASET_PATH = "./data/dataset/click_stream_validation.pth"
TEST_FINAL_DATASET_PATH = "./data/dataset/click_stream_test.pth"

ARTICLES_TO_REMOVE_PATH = "./data/processed/articles_to_remove.txt"
SELECTED_ARTICLES = "./data/processed/selected_articles.csv"

SOURCE_ARTICLE_COLUMN = "source_article"
TARGET_ARTICLE_COLUMN = "target_article"

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def remove_empty_articles(tokenized_articles, dataset):
    return dataset[
        dataset[SOURCE_ARTICLE_COLUMN].isin(tokenized_articles["article"])
        & dataset[TARGET_ARTICLE_COLUMN].isin(tokenized_articles["article"])
    ]


def run():
    tokenized_articles = pd.read_csv("./data/processed/wiki_articles_english_complete.csv")

    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    train_dataset = remove_empty_articles(tokenized_articles, train_dataset)

    validation_dataset = pd.read_csv(VALIDATION_DATASET_PATH)
    validation_dataset = remove_empty_articles(tokenized_articles, validation_dataset)

    test_dataset = pd.read_csv(TEST_DATASET_PATH)
    test_dataset = remove_empty_articles(tokenized_articles, test_dataset)

    click_stream_train = ClickStreamDataset(train_dataset)
    click_stream_validation = ClickStreamDataset(validation_dataset)
    click_stream_test = ClickStreamDataset(test_dataset)

    logger.info("Datasets split. Starting saving them")

    torch.save(click_stream_train, TRAIN_FINAL_DATASET_PATH)
    torch.save(click_stream_validation, VALIDATION_FINAL_DATASET_PATH)
    torch.save(click_stream_test, TEST_FINAL_DATASET_PATH)

    logger.info(
        f"Datasets successfully saved. \n"
        f"Train size: {len(train_dataset)} \n"
        f"Validation size: {len(validation_dataset)} \n"
        f"Test size: {len(test_dataset)} \n"
    )


if __name__ == "__main__":
    run()
