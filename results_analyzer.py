import json
import os
import sys
from itertools import chain

os.chdir(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging
import math

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple
import glob
import random

import matplotlib.pyplot as plt

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
    CLEAN_MODEL_NAMES,
    COMPLETE_MODELS,
    SMASH_WORD_LEVEL,
    SMASH_MODELS,
)

WORD_COUNT_BIN = "word_count_bin"
NDCG_COLUMN = "ndcg"
MAP_COLUMN = "map"
PRECISION_COLUMN = "precision"
IS_IN_TOP_ARTICLES_COLUMN = "is_in_top_articles"
K_COLUMN = "k"

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

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

SMASH_HATCH = "//"
DOC2VEC_HATCH = "X"
WIKIPEDIA2VEC_HATCH = "."

system_styles = {
    "doc2vec_siamese": dict(color="#b2abd2", hatch=DOC2VEC_HATCH),
    "wikipedia2vec_siamese": dict(color="#e66101", hatch=WIKIPEDIA2VEC_HATCH),
    "smash_paragraph_level": dict(color="#abdda4"),
    "smash_sentence_level": dict(color="#fdae61"),
    "smash_word_level": dict(color="#2b83ba"),
}

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE + 1)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE + 1)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE + 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc("pdf", fonttype=42)
plt.rc("ps", fonttype=42)

plt.rc("text", usetex=False)
plt.rc("font", family="serif")


class ResultsAnalyzer:
    def __init__(self):
        self.results = self.build_models_results()
        self.source_articles = self.results[SOURCE_ARTICLE_COLUMN].unique().tolist()

        self.top_articles = self.build_top_n_matrix_by_article()

        self.predictions_by_model = None
        self.predictions_by_model_and_article = None
        self.results_by_model_and_article = None

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

    def get_article_results(
        self, article: str = None, model: str = SMASH_WORD_LEVEL
    ) -> None:
        """Plots a table with the actual and predicted top 10 articles for a selected article
         and model

        Parameters
        ----------
        article : str
            Source article selected to get the actual and predicted top 10 articles
        model : str
            Model selected to get the predicted top 10 articles

        Returns
        -------
        None

        """
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

        print(f"Source article: {article}")

        print(results_table[["actual click rate", model]])

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
        if self.results_by_model_and_article:
            return self.results_by_model_and_article

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
        results_by_model_and_article = self.get_results_by_model_and_article(
            predictions_df, k=10
        )

        results_by_model_and_article = results_by_model_and_article.pivot(
            index=SOURCE_ARTICLE_COLUMN, columns=MODEL_COLUMN, values=NDCG_COLUMN
        ).reset_index()

        results_by_model_and_article = results_by_model_and_article.merge(
            articles_features_df,
            left_on=[SOURCE_ARTICLE_COLUMN],
            right_on=[ARTICLE_COLUMN],
        ).drop(columns=[ARTICLE_COLUMN])

        self.results_by_model_and_article = results_by_model_and_article

        return self.results_by_model_and_article

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

    def get_performance_figure(
        self,
        models: List[str],
        feature_column: str,
        x_label: str,
        figsize: Tuple[int, int] = (13, 6),
        legend_columns_count: int = 3,
        buckets_count: int = 8,
        save_file_name: str = None,
    ) -> None:
        """Plots a performance figure with the results of each model according
        to the features of the source articles

        Parameters
        ----------
        models : List[str]
            List with the models that will be displayed in the plot
        feature_column : str
            Column with the feature that will be used to group the articles
        x_label : str
            Label to be displayed in the x-axis
        figsize : int
            Size of the figure
        legend_columns_count : int
            Number of columns in the legend
        buckets_count : int
            Number of groups
        save_file_name : str
            Path to save the figures. Will not be saved if ommited

        Returns
        -------
        None
        """

        results = self.calculate_statistics_per_article()

        bin_column = f"{feature_column}_bin"
        bins = pd.qcut(results[feature_column], q=buckets_count)

        results[bin_column] = bins
        result_by_model = results.groupby([bin_column]).mean()[models]

        fig = plt.figure(figsize=figsize)

        ax = result_by_model.plot(
            kind="bar",
            ax=fig.gca(),
            rot=0,
            width=0.7,
            alpha=0.9,
            edgecolor=["black"],
        )

        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75]
        )

        # Formats the bars
        for container in ax.containers:
            container_system = container.get_label()

            style = system_styles[container_system]
            for patch in container.patches:
                if "color" in style:
                    patch.set_color(style["color"])
                if "hatch" in style:
                    patch.set_hatch(style["hatch"])
                if "linewidth" in style:
                    patch.set_linewidth(style["linewidth"])
                if "edgecolor" in style:
                    patch.set_edgecolor(style["edgecolor"])
                else:
                    patch.set_edgecolor("black")

        model_names = [CLEAN_MODEL_NAMES[model] for model in models]

        ax.legend(
            model_names,
            ncol=legend_columns_count,
            loc="upper center",
            fancybox=True,
            shadow=False,
            bbox_to_anchor=(0.5, 1.2),
        )

        # Formats the x label as "(lower, upper]"
        ax.set_xticklabels(
            [f"({int(i.left)}, {int(i.right)}]" for i in bins.cat.categories]
        )

        y_label = "NDCG@10"
        ax.set_xlabel(x_label % len(result_by_model))
        ax.set_ylabel(y_label)

        if save_file_name:
            save_file_path = f"./results/figures/{save_file_name}.png"
            pdf_dpi = 300

            #         logger.info(f"Saved to {save_file_path}")
            plt.savefig(save_file_path, bbox_inches="tight", dpi=pdf_dpi)

        plt.show()

    def get_performance_by_model(self, selected_models: List[str]) -> None:
        """Prints a table with the performance results by model

        Parameters
        ----------
        selected_models : List[str]
            List with the models to be plotted in the table

        Returns
        -------
        None
        """
        results_per_model = self.calculate_statistics_per_model()

        print(self._get_clean_results(results_per_model, selected_models))

    @staticmethod
    def _get_clean_results(
        model_results: pd.DataFrame, selected_models: List[str]
    ) -> pd.DataFrame:
        """Cleans the results and formats the table

        Parameters
        ----------
        model_results : pd.DataFrame
            Pandas DataFrame with the results by model
        selected_models : List[str]
            List with the models to be plotted in the table

        Returns
        -------
        pd.DataFrame
        """
        clean_results = model_results[
            model_results[MODEL_COLUMN].isin(selected_models)
        ].copy()
        clean_results[MODEL_COLUMN] = clean_results["model"].map(CLEAN_MODEL_NAMES)
        clean_results.sort_values(MODEL_COLUMN, inplace=True)

        clean_results.columns = ["Model", "NDCG@10", "MAP@10", "Precision@10"]

        return clean_results

    def get_performance_different_k(
        self, models: List[str], ks: List[int] = None, metric_column: str = NDCG_COLUMN
    ) -> None:
        """Prints a table with the performance results according to different k

        Parameters
        ----------
        models : List[str]
            List with the models to be plotted in the table
        ks : List[int]
            List with the ks to be plotted in the table
        metric_column : str
            Metric that will be used to display the results. Default `ndcg`

        Returns
        -------
        None
        """
        results_different_k = self.calculate_statistics_per_model_different_k(ks)

        performance_different_k = (
            results_different_k[results_different_k[MODEL_COLUMN].isin(models)]
            .pivot(index=MODEL_COLUMN, columns=K_COLUMN, values=metric_column)
            .reset_index()
        )
        performance_different_k[MODEL_COLUMN] = performance_different_k[
            MODEL_COLUMN
        ].map(CLEAN_MODEL_NAMES)
        performance_different_k.sort_values(MODEL_COLUMN, inplace=True)

        print(performance_different_k)


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
    _results.get_performance_figure(
        SMASH_MODELS,
        SENTENCE_COUNT_COLUMN,
        "Text length as sentence count (%s equal-sized buckets)",
    )
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
