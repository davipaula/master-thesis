import sys
import os

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging
import torch
import pandas as pd
from data_structure.click_stream_dataset import ClickStreamDataset
from utils.constants import WIKI_ARTICLES_DATASET_PATH

TRAIN_FINAL_DATASET_PATH = "./data/dataset/click_stream_train.pth"
VALIDATION_FINAL_DATASET_PATH = "./data/dataset/click_stream_validation.pth"
TEST_FINAL_DATASET_PATH = "./data/dataset/click_stream_test.pth"

TRAIN_FINAL_DATASET_CSV_PATH = "./data/processed/train.csv"
VALIDATION_FINAL_DATASET_CSV_PATH = "./data/processed/validation.csv"
TEST_FINAL_DATASET_CSV_PATH = "./data/processed/test.csv"

ARTICLES_TO_REMOVE_PATH = "./data/processed/articles_to_remove.txt"
SELECTED_ARTICLES = "./data/processed/selected_articles.csv"

SOURCE_ARTICLE_COLUMN = "source_article"
TARGET_ARTICLE_COLUMN = "target_article"

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def remove_empty_articles(tokenized_articles, dataset):
    return dataset[
        dataset[SOURCE_ARTICLE_COLUMN].isin(tokenized_articles["article"])
        & dataset[TARGET_ARTICLE_COLUMN].isin(tokenized_articles["article"])
    ]


def save_dataset(dataset, output_path):
    dataset.to_csv(output_path + ".csv", index=False)
    prepared_dataset = ClickStreamDataset(dataset)
    torch.save(prepared_dataset, output_path)


def run():
    tokenized_articles = pd.read_csv(WIKI_ARTICLES_DATASET_PATH)

    train_dataset = pd.read_csv(TRAIN_FINAL_DATASET_CSV_PATH)
    validation_dataset = pd.read_csv(VALIDATION_FINAL_DATASET_CSV_PATH)
    test_dataset = pd.read_csv(TEST_FINAL_DATASET_CSV_PATH)

    logger.info(f"Original train dataset {len(train_dataset)}")
    train_dataset = remove_empty_articles(tokenized_articles, train_dataset)
    save_dataset(train_dataset, TRAIN_FINAL_DATASET_PATH)
    logger.info(f"Filtered train dataset {len(train_dataset)}")

    logger.info(f"Original validation dataset {len(validation_dataset)}")
    validation_dataset = remove_empty_articles(tokenized_articles, validation_dataset)
    save_dataset(validation_dataset, VALIDATION_FINAL_DATASET_PATH)
    logger.info(f"Filtered validation dataset {len(validation_dataset)}")

    logger.info(f"Original test dataset {len(test_dataset)}")
    test_dataset = remove_empty_articles(
        tokenized_articles, test_dataset
    ).drop_duplicates()
    save_dataset(test_dataset, TEST_FINAL_DATASET_PATH)
    logger.info(f"Filtered test dataset {len(test_dataset)}")


if __name__ == "__main__":
    run()
