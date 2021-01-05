"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
from typing import List

import numpy as np
import pandas as pd


def average_precision_at_k(predictions: List[bool], k: int) -> float:
    """Returns average precision@k

    Parameters
    ----------
    predictions : List[bool]
        List with the relevance of predictions made by the model
    k : int
        @k position to calculate the metric

    Returns
    -------
    float
    """
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


def precision_at_k(predictions: List[bool], k: int) -> float:
    """Returns the precision@k

    Parameters
    ----------
    predictions : List[bool]
        List with the relevance of predictions made by the model
    k : int
        @k position to calculate the metric
    Returns
    -------
    float
    """
    correct_predictions = sum(predictions[:k])

    return correct_predictions / len(predictions[:k])


def calculate_precision(predictions: List[bool]) -> float:
    """

    Parameters
    ----------
    predictions : List[bool]
        List with the relevance of predictions made by the model

    Returns
    -------
    float
    """
    running_sum = 0
    correct_predictions = 0

    for prediction_index, prediction in enumerate(predictions, start=1):
        if prediction is True:
            correct_predictions += 1
            running_sum += correct_predictions / float(prediction_index)

    return running_sum / correct_predictions


def calculate_idcg_at_k_by_article(k: int) -> np.ndarray:
    """Calculates IDCG at k.

    Parameters
    ----------
    k : int
        @k parameter to calculate IDCG

    Returns
    -------
    np.ndarray
    """
    # As this is a ranking problem, the top k items will be relevant
    perfect_predictions = pd.Series([True] * k)

    return calculate_dcg_at_k_by_article(perfect_predictions, k)


def get_ndcg_at_k_by_article(predictions: pd.Series, k: int) -> np.ndarray:
    """Calculates the NDCG at k for the predictions

    Parameters
    ----------
    predictions : pd.Series
        Series with predictions for a source article
    k : int
        @k parameter to calculate the NDCG

    Returns
    -------
    np.ndarray
    """
    np.seterr(divide="ignore", invalid="ignore")
    dcg_at_k = calculate_dcg_at_k_by_article(predictions, k)

    # All source articles have 10 most visited target articles, so the perfect rank would be
    # [True, True, True, True, True, True, True, True, True, True]
    idcg_at_k = calculate_idcg_at_k_by_article(len(predictions[:k]))

    if idcg_at_k == 0:
        return np.array([0])

    return np.nan_to_num(dcg_at_k / idcg_at_k)


def calculate_dcg_at_k_by_article(predictions: pd.Series, k: int) -> np.ndarray:
    """Calculates DCG at k for the predictions

    Parameters
    ----------
    predictions : pd.Series
        Series with predictions for a source article
    k : int
        @k parameter to calculate the DCG

    Returns
    -------
    np.ndarray
    """
    article_predictions_at_k = predictions[:k]

    dcg = (np.power(2, article_predictions_at_k[1:]) - 1) / np.log2(
        np.arange(3, article_predictions_at_k.size + 2)
    )

    dcg = np.concatenate(([float(article_predictions_at_k.iloc[0])], dcg))

    return np.sum(dcg)
