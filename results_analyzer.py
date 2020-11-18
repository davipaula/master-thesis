import os
import sys

from utils.constants import (
    NDCG_COLUMN,
    MAP_COLUMN,
    PRECISION_COLUMN,
    IS_IN_TOP_ARTICLES_COLUMN,
    K_COLUMN,
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging

import pandas as pd

from tqdm import tqdm
from typing import List, Tuple
import glob
import random

import matplotlib.pyplot as plt

from src.utils.constants import (
    RESULT_FILE_COLUMNS_NAMES,
    TARGET_ARTICLE_COLUMN,
    SOURCE_ARTICLE_COLUMN,
    MODEL_COLUMN,
    ACTUAL_CLICK_RATE_COLUMN,
    PREDICTED_CLICK_RATE_COLUMN,
    ARTICLE_COLUMN,
    CLEAN_MODEL_NAMES,
    SMASH_WORD_LEVEL,
)

from src.utils.metrics_calculator import (
    get_ndcg_at_k_by_article,
    average_precision_at_k,
    precision_at_k,
)

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
        self.results = self._build_models_results()
        self.source_articles = self.results[SOURCE_ARTICLE_COLUMN].unique().tolist()

        self.top_articles = self._get_top_n_actual_articles()

        self.predictions_by_model = None
        self.predictions_by_model_and_article = None
        self.results_by_model_and_article = None

    @staticmethod
    def _build_models_results() -> pd.DataFrame:
        """Reads the results files and returns a pd.DataFrame with all results

        Returns
        -------
        pd.DataFrame
        """
        results_files = [file for file in glob.glob(f"{BASE_RESULTS_PATH}*.csv")]

        results = pd.DataFrame(columns=RESULT_FILE_COLUMNS_NAMES)

        for file_path in results_files:
            results = results.append(pd.read_csv(file_path))

        return results.drop_duplicates()

    def _get_top_n_predicted_by_article_and_model(
        self, model: str, n: int = 10
    ) -> pd.DataFrame:
        """Returns top n predicted target articles by source article and model

        Parameters
        ----------
        model : str
            Name of the model to return the top n predicted results
        n : int
            Number of results to be returned

        Returns
        -------
        pd.DataFrame

        """
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

    def _get_top_n_actual_articles(self, n: int = 10) -> pd.DataFrame:
        """Returns the top n actual results by article

        Parameters
        ----------
        n : int
            Number of actual results to return

        Returns
        -------
        pd.DataFrame
        """
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

    def get_sample_source_articles(self, n: int = 10):
        """Returns a random sample of source articles

        Parameters
        ----------
        n : int
            Number of sample source articles to be returned

        Returns
        -------

        """
        return self.results[SOURCE_ARTICLE_COLUMN].sample(n=n)

    def _get_models(self) -> pd.Series:
        """Returns the model names

        Returns
        -------
        pd.Series
        """
        return self.results[MODEL_COLUMN].unique()

    def _get_top_n_predictions_by_model(self, n: int = 10) -> pd.DataFrame:
        """Returns the top n predictions by model

        Parameters
        ----------
        n : int
            Top N articles to calculate the predictions

        Returns
        -------
        pandas.core.frame.DataFrame

        """
        if self.predictions_by_model is not None:
            return self.predictions_by_model

        models = self._get_models()

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
                self._get_top_n_predicted_by_article_and_model(model, n)
            )

        self.predictions_by_model = results

        return self.predictions_by_model

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

        predictions = self._get_top_n_predictions_by_model()

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

    def calculate_metrics_by_model(self) -> pd.DataFrame:
        """Calculates the result metrics by model

        Returns
        -------
        pd.DataFrame

        """
        predictions_df = self._get_top_n_predictions_by_model()

        logger.info("Calculating result metrics by model")
        results_by_model_and_article = self._get_results_by_model_and_article(
            predictions_df, k=10
        )

        return results_by_model_and_article.groupby(MODEL_COLUMN).mean().reset_index()

    def calculate_metrics_by_model_different_k(
        self, ks: List[int] = None
    ) -> pd.DataFrame:
        """Calculates the result metrics by model with different `ks`

        Parameters
        ----------
        ks : List[int]
            List of `ks` to calculate the metric results

        Returns
        -------
        pd.DataFrame

        """
        if not ks:
            ks = [1, 3, 5, 10, 20]

        results = pd.DataFrame(
            columns={MODEL_COLUMN, NDCG_COLUMN, MAP_COLUMN, PRECISION_COLUMN, K_COLUMN}
        )

        logger.info("Calculating results by model")
        for k in ks:
            self.top_articles = self._get_top_n_actual_articles(n=k)
            self.predictions_by_model = None
            predictions_df = self._get_top_n_predictions_by_model(n=k)

            results_by_model_and_article = self._get_results_by_model_and_article(
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

    def calculate_metrics_by_article(self) -> pd.DataFrame:
        """Calculate the result metrics by article

        Returns
        -------
        pd.DataFrame
        """
        if self.results_by_model_and_article:
            return self.results_by_model_and_article

        articles_features_df = pd.read_csv("./data/article_features.csv")

        logger.info("Getting predictions by model")
        predictions_df = self._get_top_n_predictions_by_model()

        predictions_df = predictions_df.merge(
            articles_features_df,
            left_on=[SOURCE_ARTICLE_COLUMN],
            right_on=[ARTICLE_COLUMN],
        ).drop(columns=[ARTICLE_COLUMN])

        logger.info("Calculating results by model")
        results_by_model_and_article = self._get_results_by_model_and_article(
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

    @staticmethod
    def _get_results_by_model_and_article(
        predictions_df: pd.DataFrame, k: int, selected_models: List[str] = None
    ) -> pd.DataFrame:
        """Returns the results for each model and article

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Predictions by model and article
        k : int
            k used to calculate the `@k` metrics
        selected_models : List[str]
            Models to calculate the results

        Returns
        -------
        pd.DataFrame

        """
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
                ] = get_ndcg_at_k_by_article(source_article_predictions, k=k)

                results_by_model_and_article.loc[
                    (
                        (
                            results_by_model_and_article[SOURCE_ARTICLE_COLUMN]
                            == source_article
                        )
                        & (results_by_model_and_article[MODEL_COLUMN] == model)
                    ),
                    MAP_COLUMN,
                ] = average_precision_at_k(source_article_predictions, k=k)

                results_by_model_and_article.loc[
                    (
                        (
                            results_by_model_and_article[SOURCE_ARTICLE_COLUMN]
                            == source_article
                        )
                        & (results_by_model_and_article[MODEL_COLUMN] == model)
                    ),
                    PRECISION_COLUMN,
                ] = precision_at_k(source_article_predictions, k=k)

        return results_by_model_and_article

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
        figsize : Tuple[int, int]
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

        results = self.calculate_metrics_by_article()

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
        results_per_model = self.calculate_metrics_by_model()

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
        results_different_k = self.calculate_metrics_by_model_different_k(ks)

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
