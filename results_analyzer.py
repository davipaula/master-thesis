import json
import os
import sys
from itertools import chain

K_COLUMN = "k"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging
import math

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
import glob
import random

from database import ArticlesDatabase
from utils.constants import (
    RESULT_FILE_COLUMNS_NAMES,
    TARGET_ARTICLE_COLUMN,
    SOURCE_ARTICLE_COLUMN,
    MODEL_COLUMN,
    ACTUAL_CLICK_RATE_COLUMN,
    PREDICTED_CLICK_RATE_COLUMN,
    ARTICLE_COLUMN,
    OUT_LINKS_COUNT_COLUMN,
    IN_LINKS_COUNT_COLUMN,
    PARAGRAPH_COUNT_COLUMN,
    SENTENCE_COUNT_COLUMN,
)

WORD_COUNT_BIN = "word_count_bin"
NDCG_COLUMN = "ndcg"
MAP_COLUMN = "map"
PRECISION_COLUMN = "precision"
IS_IN_TOP_ARTICLES_COLUMN = "is_in_top_articles"

BASE_RESULTS_PATH = "./results/test/"
DOC2VEC_RESULTS_PATH = BASE_RESULTS_PATH + "results_doc2vec_level_test.csv"
DOC2VEC_NO_SIGMOID_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_doc2vec_no_sigmoid_test.csv"
)
DOC2VEC_COSINE_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_doc2vec_cosine_level_test.csv"
)
WIKIPEDIA2VEC_RESULTS_PATH = BASE_RESULTS_PATH + "results_wikipedia2vec_base_test.csv"
WIKIPEDIA2VEC_NO_SIGMOID_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_wikipedia2vec_no_sigmoid_test.csv"
)
WIKIPEDIA2VEC_COSINE_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_wikipedia2vec_cosine_level_test.csv"
)
SMASH_RNN_WORD_LEVEL_RESULTS_PATH = BASE_RESULTS_PATH + "results_word_level_base.csv"
SMASH_RNN_SENTENCE_LEVEL_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_sentence_level_base.csv"
)
SMASH_RNN_PARAGRAPH_LEVEL_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_paragraph_level_base.csv"
)
SMASH_RNN_WORD_LEVEL_NO_SIGMOID_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_word_level_no_sigmoid.csv"
)
SMASH_RNN_SENTENCE_LEVEL_NO_SIGMOID_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_sentence_level_no_sigmoid.csv"
)
SMASH_RNN_PARAGRAPH_LEVEL_NO_SIGMOID_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_paragraph_level_no_sigmoid.csv"
)
SMASH_RNN_WORD_LEVEL_200D_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_word_level_200d.csv"
)
SMASH_RNN_SENTENCE_LEVEL_200D_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_sentence_level_200d.csv"
)
SMASH_RNN_PARAGRAPH_LEVEL_200D_RESULTS_PATH = (
    BASE_RESULTS_PATH + "results_paragraph_level_200d.csv"
)

# For debugging purposes only
BASE_VALIDATION_RESULTS_PATH = "./results/"
SMASH_RNN_WORD_LEVEL_VALIDATION_RESULTS_PATH = (
    BASE_VALIDATION_RESULTS_PATH + "results_word_level_validation.csv"
)
SMASH_RNN_SENTENCE_LEVEL_VALIDATION_RESULTS_PATH = (
    BASE_VALIDATION_RESULTS_PATH + "results_sentence_level_validation.csv"
)
SMASH_RNN_PARAGRAPH_LEVEL_VALIDATION_RESULTS_PATH = (
    BASE_VALIDATION_RESULTS_PATH + "results_paragraph_level_validation.csv"
)

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ResultsAnalyzer:
    def __init__(self):
        self.results = self.build_models_results()
        self.source_articles = self.results[SOURCE_ARTICLE_COLUMN].unique().tolist()

        self.top_articles = self.build_top_n_matrix_by_article()

        self.predictions_by_model = None
        self.predictions_by_model_and_article = None

    def build_models_results(self):
        results_files = [file for file in glob.glob(f"{BASE_RESULTS_PATH}*.csv")]

        results = pd.DataFrame(columns=RESULT_FILE_COLUMNS_NAMES)

        for file_path in results_files:
            results = results.append(pd.read_csv(file_path))

        return results.drop_duplicates()

    def get_top_n_predicted_by_article_and_model(self, model: str, n=10):
        model_results = self.results[(self.results[MODEL_COLUMN] == model)]
        model_results = (
            model_results.sort_values(PREDICTED_CLICK_RATE_COLUMN, ascending=False)
            .groupby(SOURCE_ARTICLE_COLUMN)
            .head(n)
        )
        model_results[IS_IN_TOP_ARTICLES_COLUMN] = False

        source_articles = set(self.results[SOURCE_ARTICLE_COLUMN])

        for source_article in source_articles:
            actual_top_articles = self.top_articles[
                self.top_articles[SOURCE_ARTICLE_COLUMN] == source_article
            ][TARGET_ARTICLE_COLUMN].unique()

            model_results.loc[
                (model_results[SOURCE_ARTICLE_COLUMN] == source_article)
                & (model_results[TARGET_ARTICLE_COLUMN].isin(actual_top_articles)),
                IS_IN_TOP_ARTICLES_COLUMN,
            ] = True

        return model_results

    def build_top_n_matrix_by_article(self, n=10):
        first_model = self.results["model"].unique()[0]
        actual_results = self.results[self.results["model"] == first_model].sort_values(
            by=[SOURCE_ARTICLE_COLUMN, ACTUAL_CLICK_RATE_COLUMN, TARGET_ARTICLE_COLUMN],
            ascending=[True, False, True],
        )

        top_n_results = (
            actual_results.groupby(SOURCE_ARTICLE_COLUMN)
            .head(n)
            .drop([ACTUAL_CLICK_RATE_COLUMN], axis=1)
        )

        top_n_results[MODEL_COLUMN] = "actual click rate"

        return top_n_results

    def get_sample_source_articles(self, n=10):
        return self.results[SOURCE_ARTICLE_COLUMN].sample(n=n)

    def get_models(self):
        return self.results[MODEL_COLUMN].unique()

    def build_validation_models_results(self):
        # For debugging purposes only
        columns_names_validation = [
            SOURCE_ARTICLE_COLUMN,
            TARGET_ARTICLE_COLUMN,
            ACTUAL_CLICK_RATE_COLUMN,
            PREDICTED_CLICK_RATE_COLUMN,
        ]
        __models_validation = {
            "smash_rnn_word_level": SMASH_RNN_WORD_LEVEL_VALIDATION_RESULTS_PATH,
            "smash_rnn_sentence_level": SMASH_RNN_SENTENCE_LEVEL_VALIDATION_RESULTS_PATH,
            "smash_rnn_paragraph_level": SMASH_RNN_PARAGRAPH_LEVEL_VALIDATION_RESULTS_PATH,
        }
        results_validation = pd.DataFrame(columns=columns_names_validation)
        for model_path in __models_validation.values():
            results_validation = results_validation.append(pd.read_csv(model_path))

        return results_validation

    @staticmethod
    def calculate_precision(predictions: List[bool]):
        running_sum = 0
        correct_predictions = 0

        for prediction_index, prediction in enumerate(predictions, start=1):
            if prediction is True:
                correct_predictions += 1
                running_sum += correct_predictions / float(prediction_index)

        return running_sum / correct_predictions

    @staticmethod
    def precision_at_k(predictions: List[bool], k: int):
        correct_predictions = sum(predictions[:k])

        return correct_predictions / len(predictions[:k])

    @staticmethod
    def average_precision_at_k(predictions: List[bool], k: int) -> float:
        predictions_at_k = predictions[:k]

        running_sum = 0
        correct_predictions = 0

        for prediction_index, prediction in enumerate(predictions_at_k, start=1):
            if prediction is True:
                correct_predictions += 1
                running_sum += correct_predictions / float(prediction_index)

        if correct_predictions == 0:
            return 0

        return running_sum / correct_predictions

    def get_predictions_by_model(self, n=10):
        if self.predictions_by_model is not None:
            return self.predictions_by_model

        models = self.get_models()

        results = pd.DataFrame(
            columns=[
                MODEL_COLUMN,
                SOURCE_ARTICLE_COLUMN,
                TARGET_ARTICLE_COLUMN,
                ACTUAL_CLICK_RATE_COLUMN,
                PREDICTED_CLICK_RATE_COLUMN,
                IS_IN_TOP_ARTICLES_COLUMN,
            ]
        )
        logger.info("Aggregating predictions for each model")
        for model in tqdm(models):
            results = results.append(
                self.get_top_n_predicted_by_article_and_model(model, n)
            )

        self.predictions_by_model = results

        return self.predictions_by_model

    def calculate_idcg(self, predictions):
        sorted_predictions = [
            sorted(article_predictions, reverse=True)
            for article_predictions in predictions
        ]

        return self.calculate_dcg(sorted_predictions)

    def calculate_idcg_at_k_by_article(self, k):
        perfect_predictions = pd.Series([True] * k)

        return self.calculate_dcg_at_k_by_article(perfect_predictions, k)

    @staticmethod
    def calculate_dcg_at_k_by_article(article_predictions, k):
        article_predictions_at_k = article_predictions[:k]

        dcg = (np.power(2, article_predictions_at_k[1:]) - 1) / np.log2(
            np.arange(3, article_predictions_at_k.size + 2)
        )

        dcg = np.concatenate(([float(article_predictions_at_k.iloc[0])], dcg))

        return np.sum(dcg)

    @staticmethod
    def calculate_dcg(predictions: List[List[bool]]):
        cumulative_gain_list = []

        for article_predictions in predictions:
            article_cumulative_gain = []
            for i, article_prediction in enumerate(article_predictions, 1):
                article_cumulative_gain.append(
                    (2 ** article_prediction - 1) / (math.log2(i + 1))
                )

            cumulative_gain_list.append(sum(article_cumulative_gain))

        return cumulative_gain_list

    def calculate_ndcg(self, predictions):
        np.seterr(divide="ignore", invalid="ignore")
        dcg = np.array(self.calculate_dcg(predictions))
        idcg = np.array(self.calculate_idcg(predictions))
        ndcg = np.nan_to_num(dcg / idcg)

        ndcg = np.mean(ndcg)

        return ndcg

    def get_article_results(self, article: str = None) -> pd.DataFrame:
        if article is None:
            article = random.choice(self.source_articles)

        predictions = self.get_predictions_by_model()

        article_results = predictions[predictions[SOURCE_ARTICLE_COLUMN] == article]

        article_actual_results = self.top_articles[
            self.top_articles[SOURCE_ARTICLE_COLUMN] == article
        ]

        article_results = article_results.append(article_actual_results)

        results_table = pd.concat(
            {
                key: value.reset_index(drop=True)
                for key, value in article_results.groupby(MODEL_COLUMN)[
                    TARGET_ARTICLE_COLUMN
                ]
            },
            axis=1,
        )

        return results_table

    def calculate_statistics_per_model(self) -> pd.DataFrame:
        predictions_df = pd.DataFrame.from_dict(self.get_predictions_by_model())

        logger.info("Calculating results by model")
        results_by_model_and_article = self.get_results_by_model_and_article(
            predictions_df, k=10
        )

        return results_by_model_and_article.groupby(MODEL_COLUMN).mean().reset_index()

    def calculate_statistics_per_model_different_k(self, ks: List[int] = None):

        if not ks:
            ks = [1, 3, 5, 10, 20]

        results = pd.DataFrame(
            columns={MODEL_COLUMN, NDCG_COLUMN, MAP_COLUMN, PRECISION_COLUMN, K_COLUMN}
        )

        logger.info("Calculating results by model")
        for k in ks:
            self.top_articles = self.build_top_n_matrix_by_article(n=k)
            self.predictions_by_model = None
            predictions_df = self.get_predictions_by_model(n=k)

            results_by_model_and_article = self.get_results_by_model_and_article(
                predictions_df, k=k
            )

            model_results = (
                results_by_model_and_article.groupby(MODEL_COLUMN).mean().reset_index()
            )

            model_results[K_COLUMN] = k

            results = results.append(model_results, ignore_index=True)

        columns_order = [
            MODEL_COLUMN,
            K_COLUMN,
            NDCG_COLUMN,
            MAP_COLUMN,
            PRECISION_COLUMN,
        ]

        results = results[columns_order]

        return results

    def calculate_statistics_per_article(self):
        db = ArticlesDatabase()

        test_articles = list(
            set(
                self.results[SOURCE_ARTICLE_COLUMN].to_list()
                + self.results[TARGET_ARTICLE_COLUMN].to_list()
            )
        )

        tokenized_words_df = pd.read_csv("./data/articles_length.csv")

        # Workaround to normalize the word count
        word_count_df = pd.read_csv("./data/articles_word_count.csv")

        tokenized_words_df = tokenized_words_df.merge(
            word_count_df, on=[ARTICLE_COLUMN]
        )

        tokenized_words_df["missing_words_percentage"] = 1 - (
            tokenized_words_df["tokenized_word_count"]
            / tokenized_words_df["word_count"]
        )

        logger.info("Getting features from DB")
        articles_features_df = pd.DataFrame.from_dict(
            db.get_features_from_articles(test_articles),
        )

        articles_features_df = articles_features_df.rename(
            columns={
                0: ARTICLE_COLUMN,
                1: OUT_LINKS_COUNT_COLUMN,
                2: IN_LINKS_COUNT_COLUMN,
                3: PARAGRAPH_COUNT_COLUMN,
                4: SENTENCE_COUNT_COLUMN,
            }
        )

        articles_features_df = articles_features_df.merge(
            tokenized_words_df, on=[ARTICLE_COLUMN]
        )

        logger.info("Getting predictions by model")
        predictions_df = pd.DataFrame.from_dict(self.get_predictions_by_model())

        predictions_df = predictions_df.merge(
            articles_features_df,
            left_on=[SOURCE_ARTICLE_COLUMN],
            right_on=[ARTICLE_COLUMN],
        ).drop(columns=[ARTICLE_COLUMN])

        logger.info("Calculating results by model")
        ndcg_by_model_and_article = self.get_results_by_model_and_article(
            predictions_df, k=10
        )

        ndcg_by_model_and_article = ndcg_by_model_and_article.pivot(
            index=SOURCE_ARTICLE_COLUMN, columns=MODEL_COLUMN, values=NDCG_COLUMN
        ).reset_index()

        ndcg_by_model_and_article = ndcg_by_model_and_article.merge(
            articles_features_df,
            left_on=[SOURCE_ARTICLE_COLUMN],
            right_on=[ARTICLE_COLUMN],
        ).drop(columns=[ARTICLE_COLUMN])

        return ndcg_by_model_and_article

    def get_results_by_model_and_article(
        self, predictions_df, k: int, selected_models=None
    ):
        if not selected_models:
            selected_models = predictions_df[MODEL_COLUMN].unique().tolist()

        results_by_model_and_article = (
            predictions_df[predictions_df[MODEL_COLUMN].isin(selected_models)][
                [MODEL_COLUMN, SOURCE_ARTICLE_COLUMN]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        source_articles = predictions_df[SOURCE_ARTICLE_COLUMN].unique().tolist()

        for model in selected_models:
            for source_article in tqdm(source_articles):
                source_article_predictions = predictions_df[
                    (predictions_df[SOURCE_ARTICLE_COLUMN] == source_article)
                    & (predictions_df[MODEL_COLUMN] == model)
                ][IS_IN_TOP_ARTICLES_COLUMN]

                results_by_model_and_article.loc[
                    (
                        (
                            results_by_model_and_article[SOURCE_ARTICLE_COLUMN]
                            == source_article
                        )
                        & (results_by_model_and_article[MODEL_COLUMN] == model)
                    ),
                    NDCG_COLUMN,
                ] = self.get_ndcg_at_k_by_article(source_article_predictions, k=k)

                results_by_model_and_article.loc[
                    (
                        (
                            results_by_model_and_article[SOURCE_ARTICLE_COLUMN]
                            == source_article
                        )
                        & (results_by_model_and_article[MODEL_COLUMN] == model)
                    ),
                    MAP_COLUMN,
                ] = self.average_precision_at_k(source_article_predictions, k=k)

                results_by_model_and_article.loc[
                    (
                        (
                            results_by_model_and_article[SOURCE_ARTICLE_COLUMN]
                            == source_article
                        )
                        & (results_by_model_and_article[MODEL_COLUMN] == model)
                    ),
                    PRECISION_COLUMN,
                ] = self.precision_at_k(source_article_predictions, k=k)

        return results_by_model_and_article

    def get_ndcg_at_k_by_article(self, source_article_predictions: List[bool], k: int):
        np.seterr(divide="ignore", invalid="ignore")
        dcg_at_k = self.calculate_dcg_at_k_by_article(source_article_predictions, k)

        # All source articles have 10 most visited target articles, so the perfect rank would be
        # [True, True, True, True, True, True, True, True, True, True]
        idcg_at_k = self.calculate_idcg_at_k_by_article(
            len(source_article_predictions[:k])
        )

        if idcg_at_k == 0:
            return 0

        try:
            return np.nan_to_num(dcg_at_k / idcg_at_k)

        except Exception as err:
            print(err)

            exit(1)

    def calculate_tokenized_lengths(self, article):
        tokenized_length = len(self.flatten_article(json.loads(article)))

        return tokenized_length

    def calculate_tokenized_lengths_original(self, articles):
        tokenized_length = [
            len(self.flatten_article(json.loads(article))) for article in articles
        ]

        return pd.Series(tokenized_length)

    def flatten_article(self, article):
        flatten_sentences = list(chain.from_iterable(article))
        flatten_words = list(chain.from_iterable(flatten_sentences))

        return flatten_words


if __name__ == "__main__":
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    _results = ResultsAnalyzer()
    pred = [
        [True, True, True, True, True],
        [False, False, False, False, False],
        [True, False, True, False, True],
    ]
    # print(_results.calculate_mean_average_precision_at_k(pred))
    # results = _results.calculate_statistics_per_model()
    results_per_article = _results.calculate_statistics_per_article()
    # print(results_per_article.describe())
    #
    # results_per_model = _results.calculate_statistics_per_model()
    # print(results_per_model)
    #
    # _results.get_article_results("Ireland")

    # results_per_model_2 = _results.calculate_statistics_per_model_different_k(
    #     [1, 3, 5, 10]
    # )
    # DOC2VEC_SIAMESE = "doc2vec_siamese"
    # DOC2VEC_COSINE = "doc2vec_cosine"
    # WIKIPEDIA2VEC_SIAMESE = "wikipedia2vec_siamese"
    # WIKIPEDIA2VEC_COSINE = "wikipedia2vec_cosine"
    # SMASH_WORD_LEVEL = "smash_word_level"
    # SMASH_SENTENCE_LEVEL = "smash_sentence_level"
    # SMASH_PARAGRAPH_LEVEL = "smash_paragraph_level"
    # SMASH_WORD_LEVEL_INTRODUCTION = "smash_word_level_introduction"
    # SMASH_SENTENCE_LEVEL_INTRODUCTION = "smash_sentence_level_introduction"
    # SMASH_PARAGRAPH_LEVEL_INTRODUCTION = "smash_paragraph_level_introduction"
    #
    # COMPLETE_MODELS = [
    #     DOC2VEC_SIAMESE,
    #     WIKIPEDIA2VEC_SIAMESE,
    #     SMASH_WORD_LEVEL,
    #     SMASH_SENTENCE_LEVEL,
    #     SMASH_PARAGRAPH_LEVEL,
    # ]
    #
    # nn = results_per_model_2[results_per_model_2["model"].isin(COMPLETE_MODELS)].pivot(
    #     index="model", columns="k", values="ndcg"
    # )
    # print(nn)

    # _results.get_ndcg_for_all_models()
