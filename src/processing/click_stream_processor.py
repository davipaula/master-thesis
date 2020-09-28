from ast import literal_eval

import pandas as pd
import os
import random
import torch
from datetime import datetime
import logging
import database
import numpy as np

from tqdm import tqdm

from data_structure.click_stream_dataset import ClickStreamDataset
from data_structure.click_stream_pre_processed import ClickStreamPreProcessed
from torch.utils.data import TensorDataset, random_split

from utils.constants import (
    WIKI_ARTICLES_DATASET_PATH,
    CLICK_STREAM_TRAIN_DATASET_PATH,
    CLICK_STREAM_VALIDATION_DATASET_PATH,
    CLICK_STREAM_TEST_DATASET_PATH,
    SELECTED_ARTICLES_PATH,
    AVAILABLE_TITLES_PATH,
)

NUMBER_OF_CLICKS_COLUMN = "number_of_clicks"

SOURCE_ARTICLE_COLUMN = "source_article"
TARGET_ARTICLE_COLUMN = "target_article"

ARTICLE_TOTAL_CLICKS_THRESHOLD = 10000
TARGET_CLICKS_THRESHOLD = 200

TRAIN_DATASET_SPLIT = 0.7

DATASET_SAMPLE_PERCENT = 0.1

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ClickStreamProcessor:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        random.seed(123)
        np.random.seed(123)

        self.articles_database = database.ArticlesDatabase()

        self.tokenized_articles = pd.read_csv(WIKI_ARTICLES_DATASET_PATH)

    @staticmethod
    def get_available_titles_in_wiki_articles():
        with open(AVAILABLE_TITLES_PATH) as f:
            selected_articles = f.read().splitlines()

        return selected_articles

    def filter_dataset(self, click_stream):
        click_stream = self.filter_available_titles(click_stream)
        click_stream = self.filter_number_of_clicks_threshold(click_stream)

        return click_stream

    def filter_available_titles(self, click_stream):
        available_titles = self.get_available_titles_in_wiki_articles()

        return click_stream[
            (click_stream[SOURCE_ARTICLE_COLUMN].isin(available_titles))
            & (click_stream[TARGET_ARTICLE_COLUMN].isin(available_titles))
        ].copy()

    @staticmethod
    def filter_number_of_clicks_threshold(click_stream):
        number_of_clicks_per_source_article = click_stream.groupby(
            SOURCE_ARTICLE_COLUMN
        )[NUMBER_OF_CLICKS_COLUMN].sum()
        articles_above_threshold = number_of_clicks_per_source_article[
            number_of_clicks_per_source_article > ARTICLE_TOTAL_CLICKS_THRESHOLD
        ].index

        return click_stream[
            (click_stream[SOURCE_ARTICLE_COLUMN].isin(articles_above_threshold))
            & (click_stream[NUMBER_OF_CLICKS_COLUMN] > TARGET_CLICKS_THRESHOLD)
        ]

    def run(self, batch_size=2, save_folder="./data/dataset"):
        logger.info("Beginning of dataset split")

        self.split_datasets()

    def generate_dataset_sample(self):
        _pre_processed_dataset = ClickStreamPreProcessed().dataset

        logger.info(f"Clickstream dataset original size {len(_pre_processed_dataset)}")
        logger.info("Filtering dataset")
        dataset = self.filter_dataset(_pre_processed_dataset)

        logger.info(f"Filtered clickstream dataset size {len(dataset)}")
        logger.info("Adding negative sampling")
        dataset = self.add_negative_sampling(dataset)

        logger.info(f"Finished adding negative sampling. Dataset size {len(dataset)}")

        unique_source_articles = (
            dataset[SOURCE_ARTICLE_COLUMN].drop_duplicates().reset_index(drop=True)
        )

        num_articles = len(unique_source_articles)
        sample_dataset_size = int(num_articles * DATASET_SAMPLE_PERCENT)

        selected_indices = random.sample(range(num_articles), sample_dataset_size)

        selected_articles = unique_source_articles[
            unique_source_articles.index.isin(selected_indices)
        ]

        dataset_sample = dataset[
            dataset[SOURCE_ARTICLE_COLUMN].isin(selected_articles)
        ].reset_index(drop=True)

        self.save_selected_articles_file(dataset_sample)

        return dataset_sample

    @staticmethod
    def save_selected_articles_file(dataset):
        selected_articles = pd.Series.append(
            dataset[SOURCE_ARTICLE_COLUMN].drop_duplicates(),
            dataset[TARGET_ARTICLE_COLUMN].drop_duplicates(),
        ).reset_index(drop=True)

        selected_articles.to_csv(SELECTED_ARTICLES_PATH, header=False, index=False)

    def split_datasets(self):
        random.seed(123)
        click_stream_data = self.generate_dataset_sample()

        source_articles = click_stream_data[SOURCE_ARTICLE_COLUMN].unique().tolist()

        click_stream_size = len(source_articles)
        train_dataset_size = int(click_stream_size * TRAIN_DATASET_SPLIT)
        validation_dataset_size = int((click_stream_size - train_dataset_size) / 2)
        test_dataset_size = (
            click_stream_size - train_dataset_size - validation_dataset_size
        )

        random.shuffle(source_articles)

        train_source_articles = source_articles[:train_dataset_size]
        validation_source_articles = source_articles[
            train_dataset_size : (train_dataset_size + validation_dataset_size)
        ]
        test_source_articles = source_articles[-test_dataset_size:]

        train_and_validation_source_articles = set(
            train_source_articles + validation_source_articles
        )
        test_dataset = click_stream_data[
            click_stream_data[SOURCE_ARTICLE_COLUMN].isin(test_source_articles)
            & ~click_stream_data[TARGET_ARTICLE_COLUMN].isin(
                train_and_validation_source_articles
            )
        ]

        test_all_articles = list(
            set(
                test_dataset[SOURCE_ARTICLE_COLUMN].tolist()
                + test_dataset[TARGET_ARTICLE_COLUMN].tolist()
            )
        )

        test_all_articles_and_train_source_articles = set(
            test_all_articles + train_source_articles
        )

        validation_dataset = click_stream_data[
            click_stream_data[SOURCE_ARTICLE_COLUMN].isin(validation_source_articles)
            & ~click_stream_data[TARGET_ARTICLE_COLUMN].isin(
                test_all_articles_and_train_source_articles
            )
        ]

        validation_all_articles = list(
            set(
                validation_dataset[SOURCE_ARTICLE_COLUMN].tolist()
                + validation_dataset[TARGET_ARTICLE_COLUMN].tolist()
            )
        )

        validation_and_test_all_articles = set(
            validation_all_articles + test_all_articles
        )

        train_dataset = click_stream_data[
            click_stream_data[SOURCE_ARTICLE_COLUMN].isin(train_source_articles)
            & ~click_stream_data[TARGET_ARTICLE_COLUMN].isin(
                validation_and_test_all_articles
            )
        ]

        self.normalize_and_save_dataset(train_dataset, CLICK_STREAM_TRAIN_DATASET_PATH)
        self.normalize_and_save_dataset(
            validation_dataset, CLICK_STREAM_VALIDATION_DATASET_PATH
        )
        self.normalize_and_save_dataset(test_dataset, CLICK_STREAM_TEST_DATASET_PATH)

    def add_negative_sampling(self, click_stream_dataset):
        unique_source_articles_titles = click_stream_dataset["source_article"].unique()
        logger.info("Getting wiki articles links")
        wiki_articles_links = self.articles_database.get_links_from_articles(
            unique_source_articles_titles
        )
        wiki_articles_links = pd.DataFrame(
            wiki_articles_links, columns=["article", "links"]
        )

        # wiki_articles_links["links"] = wiki_articles_links["links"].map(literal_eval)
        logger.info("Finished getting wiki articles links")
        source_articles = (
            pd.merge(
                click_stream_dataset[["source_article"]],
                wiki_articles_links[["article", "links"]],
                left_on=["source_article"],
                right_on=["article"],
            )
            .drop(["article"], axis=1)
            .drop_duplicates(subset="source_article")
        )

        source_articles["links"] = source_articles["links"].map(literal_eval)
        source_articles = source_articles.explode("links")
        visited_articles = click_stream_dataset[["source_article", "target_article"]]

        logger.info("Getting non visited articles")
        non_visited_articles = pd.merge(
            source_articles,
            visited_articles,
            left_on=["source_article", "links"],
            right_on=["source_article", "target_article"],
            how="left",
        )
        non_visited_articles = non_visited_articles[
            non_visited_articles["target_article"].isna()
        ]
        non_visited_articles = non_visited_articles.drop(["target_article"], axis=1)
        non_visited_articles.columns = ["source_article", "target_article"]

        non_visited_articles = self.filter_available_titles(non_visited_articles)

        # Gets only up to 20 negative sampling articles to reduce set size and speed up tokenization
        size = 20  # sample size
        random_function = lambda obj: obj.loc[
            np.random.RandomState(123).choice(obj.index, size), :
        ]
        non_visited_articles = non_visited_articles.groupby(
            "source_article", as_index=False
        ).apply(random_function)

        # Adds the non visited links data
        logger.info("Adding columns")
        # Remove this
        negative_sampling = non_visited_articles
        negative_sampling.insert(len(negative_sampling.columns), "number_of_clicks", 0)
        negative_sampling.insert(len(negative_sampling.columns), "click_rate", 0)

        return click_stream_dataset.append(negative_sampling, ignore_index=True)

    def get_wiki_articles_links(self, source_articles):
        return self.articles_database.get_links_from_articles(source_articles)

    def get_valid_articles(self, target_articles):
        return self.articles_database.get_valid_articles(target_articles)

    def remove_empty_articles(self, tokenized_articles, dataset):
        return dataset[
            dataset[SOURCE_ARTICLE_COLUMN].isin(tokenized_articles["article"])
            & dataset[TARGET_ARTICLE_COLUMN].isin(tokenized_articles["article"])
        ]

    def normalize_and_save_dataset(
        self, dataset_source, dataset_output_path: str
    ) -> None:
        click_stream_dataset = self.remove_empty_articles(
            self.tokenized_articles, dataset_source
        )

        click_stream_dataset.to_csv(dataset_output_path + ".csv", index=False)

        torch.save(click_stream_dataset, dataset_output_path)


if __name__ == "__main__":
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    os.chdir("/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/")

    creator = ClickStreamProcessor()
    start = datetime.now()
    logger.info("Started process")
    creator.run()
    logger.info(f"Finished process. Time elapsed: {datetime.now() - start}")
