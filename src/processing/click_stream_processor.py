from ast import literal_eval
from typing import List

import pandas as pd
import os
import random
import torch
from datetime import datetime
import logging
import database
import numpy as np

from data_structure.click_stream_pre_processed import ClickStreamPreProcessed

from utils.constants import (
    CLICK_STREAM_TRAIN_DATASET_PATH,
    CLICK_STREAM_VALIDATION_DATASET_PATH,
    CLICK_STREAM_TEST_DATASET_PATH,
    SELECTED_ARTICLES_PATH,
    AVAILABLE_TITLES_PATH,
    NUMBER_OF_CLICKS_COLUMN,
    SOURCE_ARTICLE_COLUMN,
    TARGET_ARTICLE_COLUMN,
    ARTICLE_COLUMN,
    CLICK_RATE_COLUMN,
)

LINKS_COLUMN = "links"

ARTICLE_TOTAL_CLICKS_THRESHOLD = 10000
TARGET_CLICKS_THRESHOLD = 200

TRAIN_DATASET_SPLIT = 0.8

DATASET_SAMPLE_PERCENT = 0.05

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def generate_datasets() -> None:
    """Processes click stream and generates the training, validation and evaluation datasets.

    Returns
    -------
    None

    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    random.seed(123)
    np.random.seed(123)

    logger.info("Beginning of dataset split")

    split_datasets()


def save_dataset(click_stream_dataset: pd.DataFrame, dataset_output_path: str) -> None:
    """Saves dataset in CSV and torch formats

    Parameters
    ----------
    click_stream_dataset :
    dataset_output_path :

    Returns
    -------
    None
    """
    click_stream_dataset.to_csv(dataset_output_path + ".csv", index=False)
    torch.save(click_stream_dataset, dataset_output_path)


def filter_available_titles(click_stream: pd.DataFrame) -> pd.DataFrame:
    """Filters out titles that are not available in Wikipedia dump

    Parameters
    ----------
    click_stream : pd.DataFrame
        Pre processed Wikipedia Clickstream dataset

    Returns
    -------
    pandas.core.frame.DataFrame

    """
    available_titles = get_available_titles_in_wiki_dump()

    return click_stream[
        (click_stream[SOURCE_ARTICLE_COLUMN].isin(available_titles))
        & (click_stream[TARGET_ARTICLE_COLUMN].isin(available_titles))
    ]


def filter_dataset(click_stream: pd.DataFrame) -> pd.DataFrame:
    """Executes functions to filter Clickstream dataset

    Parameters
    ----------
    click_stream : pd.DataFrame
        Pre processed Wikipedia Clickstream dataset

    Returns
    -------
    pandas.core.frame.DataFrame

    """
    click_stream = filter_available_titles(click_stream)
    click_stream = filter_number_of_clicks_threshold(click_stream)

    return click_stream


def add_negative_sampling(
    unique_source_articles: pd.Series, complete_click_stream_dataset: pd.DataFrame
) -> pd.DataFrame:
    """Filters the dataset using the source articles that meet the criteria and adds
    negative sample

    Parameters
    ----------
    unique_source_articles : pd.Series
        Unique source articles after dataset filtering
    complete_click_stream_dataset : pd.DataFrame
        Pre processed Wikipedia Clickstream dataset

    Returns
    -------
    pandas.core.frame.DataFrame

    """
    logger.info("Getting wiki articles links")
    db = database.ArticlesDatabase()
    wiki_articles_links = pd.DataFrame(
        db.get_links_from_articles(unique_source_articles),
        columns=[ARTICLE_COLUMN, LINKS_COLUMN],
    )

    filtered_dataset = filter_available_titles(complete_click_stream_dataset)

    filtered_dataset = filtered_dataset[
        complete_click_stream_dataset[SOURCE_ARTICLE_COLUMN].isin(
            unique_source_articles
        )
    ]
    logger.info("Finished getting wiki articles links")

    source_articles = (
        pd.merge(
            filtered_dataset[[SOURCE_ARTICLE_COLUMN]],
            wiki_articles_links[[ARTICLE_COLUMN, LINKS_COLUMN]],
            left_on=[SOURCE_ARTICLE_COLUMN],
            right_on=[ARTICLE_COLUMN],
        )
        .drop([ARTICLE_COLUMN], axis=1)
        .drop_duplicates(subset=SOURCE_ARTICLE_COLUMN)
    )

    source_articles[LINKS_COLUMN] = source_articles[LINKS_COLUMN].map(literal_eval)
    source_articles = source_articles.explode(LINKS_COLUMN)
    visited_articles = filtered_dataset[[SOURCE_ARTICLE_COLUMN, TARGET_ARTICLE_COLUMN]]

    logger.info("Getting non visited articles")
    non_visited_articles = get_non_visited_articles(source_articles, visited_articles)

    # Gets only 20% of non-visited articles as negative sampling articles
    # to reduce set size and speed up tokenization
    negative_sampling = non_visited_articles.groupby(SOURCE_ARTICLE_COLUMN).sample(
        frac=0.20
    )

    # Adds the non visited links data
    logger.info("Adding columns")

    negative_sampling.insert(len(negative_sampling.columns), NUMBER_OF_CLICKS_COLUMN, 0)
    negative_sampling.insert(len(negative_sampling.columns), CLICK_RATE_COLUMN, 0)

    return filtered_dataset.append(negative_sampling, ignore_index=True)


def get_non_visited_articles(
    source_articles: pd.DataFrame, visited_articles: pd.DataFrame
) -> pd.DataFrame:
    """Returns a DataFrame with all the non visited articles

    Parameters
    ----------
    source_articles : pd.DataFrame

    visited_articles : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    """
    non_visited_articles = pd.merge(
        source_articles,
        visited_articles,
        left_on=[SOURCE_ARTICLE_COLUMN, "links"],
        right_on=[SOURCE_ARTICLE_COLUMN, TARGET_ARTICLE_COLUMN],
        how="left",
    )
    non_visited_articles = non_visited_articles[
        non_visited_articles[TARGET_ARTICLE_COLUMN].isna()
    ]
    non_visited_articles = non_visited_articles.drop([TARGET_ARTICLE_COLUMN], axis=1)
    non_visited_articles.columns = [SOURCE_ARTICLE_COLUMN, TARGET_ARTICLE_COLUMN]
    non_visited_articles = filter_available_titles(non_visited_articles)

    return non_visited_articles


def prepare_dataset() -> pd.DataFrame:
    """Prepares the dataset extracted from click stream original data by:
        - filtering out articles that are not present in Wikipedia dump
        - filtering out articles that do not meet the number of clicks
        - adds negative sampling
    threshold

    Returns
    -------
    pandas.core.frame.DataFrame

    """
    _pre_processed_dataset = ClickStreamPreProcessed().dataset

    logger.info(f"Clickstream dataset original size {len(_pre_processed_dataset)}")
    logger.info("Filtering dataset")
    dataset = filter_dataset(_pre_processed_dataset)

    logger.info(f"Filtered clickstream dataset size {len(dataset)}")
    logger.info("Adding negative sampling")
    unique_source_articles = (
        dataset[SOURCE_ARTICLE_COLUMN].drop_duplicates().reset_index(drop=True)
    )

    dataset = add_negative_sampling(unique_source_articles, _pre_processed_dataset)

    logger.info(f"Finished adding negative sampling. Dataset size {len(dataset)}")

    return dataset


def split_datasets() -> None:
    """Splits datasets into training, validation and validation

    Returns
    -------
    None
    """
    random.seed(123)
    source_dataset = prepare_dataset().copy()

    click_stream_data_sum = (
        source_dataset.groupby(SOURCE_ARTICLE_COLUMN).sum().reset_index()
    )

    unique_source_articles = click_stream_data_sum[
        click_stream_data_sum[CLICK_RATE_COLUMN] > 0.95
    ][SOURCE_ARTICLE_COLUMN].drop_duplicates()
    num_source_articles = len(unique_source_articles)

    num_source_articles_sample = int(num_source_articles * DATASET_SAMPLE_PERCENT)

    num_source_articles_train = int(num_source_articles_sample * TRAIN_DATASET_SPLIT)
    num_source_articles_validation = int(
        (num_source_articles_sample - num_source_articles_train) / 2
    )
    num_source_articles_test = (
        num_source_articles_sample
        - num_source_articles_train
        - num_source_articles_validation
    )

    test_source_articles_sample = unique_source_articles.sample(
        n=num_source_articles_test, random_state=123
    )
    test_dataset = source_dataset[
        source_dataset[SOURCE_ARTICLE_COLUMN].isin(test_source_articles_sample)
    ]

    test_all_articles = set(
        test_dataset[SOURCE_ARTICLE_COLUMN].tolist()
        + test_dataset[TARGET_ARTICLE_COLUMN].tolist()
    )

    # Filter out test articles from click_stream_data for validation dataset
    source_dataset = source_dataset[
        ~source_dataset[SOURCE_ARTICLE_COLUMN].isin(test_all_articles)
        & ~source_dataset[TARGET_ARTICLE_COLUMN].isin(test_all_articles)
    ]

    click_stream_data_sum = (
        source_dataset.groupby(SOURCE_ARTICLE_COLUMN).sum().reset_index()
    )

    unique_source_articles = click_stream_data_sum[
        click_stream_data_sum[CLICK_RATE_COLUMN] > 0.95
    ][SOURCE_ARTICLE_COLUMN].drop_duplicates()

    validation_source_articles_sample = unique_source_articles.sample(
        n=num_source_articles_validation, random_state=123
    )
    validation_dataset = source_dataset[
        source_dataset[SOURCE_ARTICLE_COLUMN].isin(validation_source_articles_sample)
    ]

    validation_all_articles = set(
        validation_dataset[SOURCE_ARTICLE_COLUMN].tolist()
        + validation_dataset[TARGET_ARTICLE_COLUMN].tolist()
    )

    # Filter out test articles from click_stream_data for training dataset
    source_dataset = source_dataset[
        ~source_dataset[SOURCE_ARTICLE_COLUMN].isin(validation_all_articles)
        & ~source_dataset[TARGET_ARTICLE_COLUMN].isin(validation_all_articles)
    ]

    click_stream_data_sum = (
        source_dataset.groupby(SOURCE_ARTICLE_COLUMN).sum().reset_index()
    )

    unique_source_articles = click_stream_data_sum[
        click_stream_data_sum[CLICK_RATE_COLUMN] > 0.95
    ][SOURCE_ARTICLE_COLUMN].drop_duplicates()

    train_source_articles_sample = unique_source_articles.sample(
        n=num_source_articles_train, random_state=123
    )
    train_dataset = source_dataset[
        source_dataset[SOURCE_ARTICLE_COLUMN].isin(train_source_articles_sample)
    ]

    train_all_articles = set(
        train_dataset[SOURCE_ARTICLE_COLUMN].tolist()
        + train_dataset[TARGET_ARTICLE_COLUMN].tolist()
    )

    all_articles = train_all_articles | validation_all_articles | test_all_articles

    logger.info(f"# of unique articles {len(all_articles)}")
    logger.info(f"# rows train dataset {len(train_dataset)}")
    logger.info(f"# rows validation dataset {len(validation_dataset)}")
    logger.info(f"# rows test dataset {len(test_dataset)}")

    save_selected_articles_file(all_articles)

    save_dataset(train_dataset, CLICK_STREAM_TRAIN_DATASET_PATH)
    save_dataset(validation_dataset, CLICK_STREAM_VALIDATION_DATASET_PATH)
    save_dataset(test_dataset, CLICK_STREAM_TEST_DATASET_PATH)


def get_available_titles_in_wiki_dump() -> List[str]:
    """Gets the titles available in Wikipedia dump

    Returns
    -------
    List[str]

    """
    with open(AVAILABLE_TITLES_PATH) as f:
        selected_articles = f.read().splitlines()

    return selected_articles


def filter_number_of_clicks_threshold(click_stream: pd.DataFrame) -> pd.DataFrame:
    """Filters out the articles that are below the number of clicks threshold

    Parameters
    ----------
    click_stream : pd.DataFrame
        Pre processed Wikipedia Clickstream dataset

    Returns
    -------
    pandas.core.frame.DataFrame

    """
    number_of_clicks_per_source_article = click_stream.groupby(SOURCE_ARTICLE_COLUMN)[
        NUMBER_OF_CLICKS_COLUMN
    ].sum()
    articles_above_threshold = number_of_clicks_per_source_article[
        number_of_clicks_per_source_article > ARTICLE_TOTAL_CLICKS_THRESHOLD
    ].index

    return click_stream[
        (click_stream[SOURCE_ARTICLE_COLUMN].isin(articles_above_threshold))
        & (click_stream[NUMBER_OF_CLICKS_COLUMN] > TARGET_CLICKS_THRESHOLD)
    ]


def save_selected_articles_file(all_articles: set) -> None:
    """Saves a file with the title of all articles in training, validation and evaluation datasets

    Parameters
    ----------
    all_articles : set
        Set with the articles in training, validation and evaluation datasets

    Returns
    -------
    None
    """
    selected_articles = pd.Series(list(all_articles))

    selected_articles.to_csv(SELECTED_ARTICLES_PATH, header=False, index=False)
