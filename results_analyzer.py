import logging
import math

import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from typing import List

IS_IN_TOP_ARTICLES_COLUMN = "is_in_top_articles"

BASE_RESULTS_PATH = "./results/test/"
DOC2VEC_RESULTS_PATH = BASE_RESULTS_PATH + "results_doc2vec_level_test.csv"
WIKIPEDIA2VEC_RESULTS_PATH = BASE_RESULTS_PATH + "results_wikipedia2vec_level_test.csv"
SMASH_RNN_WORD_LEVEL_RESULTS_PATH = BASE_RESULTS_PATH + "results_word_level.csv"
SMASH_RNN_SENTENCE_LEVEL_RESULTS_PATH = BASE_RESULTS_PATH + "results_sentence_level.csv"
SMASH_RNN_PARAGRAPH_LEVEL_RESULTS_PATH = BASE_RESULTS_PATH + "results_paragraph_level.csv"

# For debugging purposes only
BASE_VALIDATION_RESULTS_PATH = "./results/"
SMASH_RNN_WORD_LEVEL_VALIDATION_RESULTS_PATH = BASE_VALIDATION_RESULTS_PATH + "results_word_level_validation.csv"
SMASH_RNN_SENTENCE_LEVEL_VALIDATION_RESULTS_PATH = (
    BASE_VALIDATION_RESULTS_PATH + "results_sentence_level_validation.csv"
)
SMASH_RNN_PARAGRAPH_LEVEL_VALIDATION_RESULTS_PATH = (
    BASE_VALIDATION_RESULTS_PATH + "results_paragraph_level_validation.csv"
)

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ResultsAnalyzer:
    def __init__(self):
        self.results = self.build_models_results()
        self.source_articles = self.results["source_article"].unique().tolist()

        self.top_articles = self.build_top_10_matrix_by_article()

    def build_models_results(self):
        # __models = {
        #     "doc2vec": DOC2VEC_RESULTS_PATH,
        #     "wikipedia2vec": WIKIPEDIA2VEC_RESULTS_PATH,
        #     "smash_rnn_word_level": SMASH_RNN_WORD_LEVEL_RESULTS_PATH,
        #     "smash_rnn_sentence_level": SMASH_RNN_SENTENCE_LEVEL_RESULTS_PATH,
        #     "smash_rnn_paragraph_level": SMASH_RNN_PARAGRAPH_LEVEL_RESULTS_PATH,
        # }

        __models = {
            "smash_rnn_word_level": SMASH_RNN_WORD_LEVEL_RESULTS_PATH,
        }

        columns_names = [
            "model",
            "source_article",
            "target_article",
            "actual_click_rate",
            "predicted_click_rate",
        ]

        results = pd.DataFrame(columns=columns_names)

        for model_path in __models.values():
            if os.path.exists(model_path):
                results = results.append(pd.read_csv(model_path))

        return results

    def get_top_10_predicted_by_article_and_model(self, source_article: str, model: str):
        n = 10

        model_results = self.results[
            (self.results["model"] == model) & (self.results["source_article"] == source_article)
        ]
        model_results = (
            model_results.sort_values("predicted_click_rate", ascending=False).groupby("source_article").head(n)
        )

        model_results[IS_IN_TOP_ARTICLES_COLUMN] = False
        actual_top_articles = self.top_articles[self.top_articles["source_article"] == source_article][
            "target_article"
        ].unique()
        model_results.loc[model_results["target_article"].isin(actual_top_articles), IS_IN_TOP_ARTICLES_COLUMN] = True

        return model_results

    def build_top_10_matrix_by_article(self):
        n = 10

        actual_results = (
            self.results[self.results["model"] == "word"]
            .sort_values(by=["source_article", "actual_click_rate"], ascending=[True, False])
            .groupby("source_article")
            .head(n)
            .drop(["actual_click_rate", "predicted_click_rate"], axis=1)
        )

        actual_results["model"] = "actual click rate"

        return actual_results

    def get_sample_source_articles(self, n=10):
        return self.results["source_article"].sample(n=n)

    def get_models(self):
        return self.results["model"].unique()

    def build_validation_models_results(self):
        # For debugging purposes only
        columns_names_validation = [
            "source_article",
            "target_article",
            "actual_click_rate",
            "predicted_click_rate",
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
        running_sum = []
        correct_predictions = 0

        for prediction_index, prediction in enumerate(predictions):
            correct_predictions += 1 if prediction is True else 0
            running_sum.append(correct_predictions / float(prediction_index + 1))

        return running_sum

    def precision_at_k(self, predictions: List[bool], k=5):
        return self.calculate_precision(predictions)[k - 1]

    def average_precision_at_k(self, predictions, k=5):
        precisions = self.calculate_precision(predictions)[: k - 1]
        average_precisions = []

        for precision_index, precision in enumerate(precisions, 1):
            if predictions[precision_index - 1] is True:
                average_precisions.append(precision)

        if average_precisions:
            result = round(sum(average_precisions) / len(average_precisions), 4)
        else:
            result = 0

        return result

    def calculate_mean_average_precision_at_k(self, predictions_by_article, k=5):
        average_precisions = []

        for predictions in predictions_by_article:
            average_precisions.append(self.average_precision_at_k(predictions, k))

        model_map = sum(average_precisions) / len(average_precisions)

        return model_map

    def get_map_for_all_models(self, k=5):
        models = self.get_models()

        predictions_by_model = {model: [] for model in models}

        logger.info("Calculating MAP for each model")
        for model in tqdm(models):
            predictions = []

            for source_article in self.source_articles:
                predictions.append(
                    self.get_top_10_predicted_by_article_and_model(source_article, model)[
                        IS_IN_TOP_ARTICLES_COLUMN
                    ].tolist()
                )

            predictions_by_model[model] = round(self.calculate_mean_average_precision_at_k(predictions, k), 4)

        return predictions_by_model

    def get_ndcg_for_all_models(self, k=5):
        models = self.get_models()

        ndcg_by_model = {model: [] for model in models}

        logger.info("Calculating NDCG for each model")
        for model in tqdm(models):
            model_predictions = []

            for source_article in self.source_articles:
                model_predictions.append(
                    self.get_top_10_predicted_by_article_and_model(source_article, model)[
                        IS_IN_TOP_ARTICLES_COLUMN
                    ].tolist()
                )

            ndcg_by_model[model] = self.calculate_ndcg(model_predictions, k)

        return ndcg_by_model

    def calculate_idcg(self, predictions):
        sorted_predictions = [sorted(article_predictions, reverse=True) for article_predictions in predictions]

        return self.calculate_dcg(sorted_predictions)

    @staticmethod
    def calculate_dcg(predictions: List[List[bool]]):
        cumulative_gain_list = []

        for article_predictions in predictions:
            article_cumulative_gain = []
            for i, article_prediction in enumerate(article_predictions, 1):
                article_cumulative_gain.append((2 ** article_prediction - 1) / (math.log2(i + 1)))

            cumulative_gain_list.append(sum(article_cumulative_gain))

        return cumulative_gain_list

    def calculate_ndcg(self, predictions, k=5):
        np.seterr(divide="ignore", invalid="ignore")
        dcg = np.array(self.calculate_dcg(predictions))
        idcg = np.array(self.calculate_idcg(predictions))
        ndcg = np.nan_to_num(dcg / idcg)

        ndcg = np.mean(ndcg)

        return ndcg


if __name__ == "__main__":
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    results = ResultsAnalyzer()
    print(results.get_ndcg_for_all_models())
