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
import argparse
from src.utils.utils import load_embeddings_from_file

from utils.constants import (
    RESULT_FILE_COLUMNS_NAMES,
    WIKI_ARTICLES_DATASET_PATH,
    TEST_DATASET_PATH,
    MODEL_FOLDER,
    SOURCE_ARTICLE_COLUMN,
    TARGET_ARTICLE_COLUMN,
    CLICK_RATE_COLUMN,
)

from utils.utils import get_word2vec_path, get_model_name

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def test(opt: argparse.Namespace) -> None:
    """Executes a test step for Smash RNN

    Parameters
    ----------
    opt : argparse.Namespace
        Model parameters. See `get_args()`

    Returns
    -------
    None
    """
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


def load_model(
    model_folder: str, model_name: str, opt: argparse.Namespace
) -> SmashRNNModel:
    """Loads the Smash RNN model to execute the test steps

    Parameters
    ----------
    model_folder : str
        Path for the folder where models are stored
    model_name : str
        Name of the model to be loaded
    opt : argparse.Namespace
        Arguments to run the test

    Returns
    -------
    SmashRNNModel

    """
    word2vec_path = get_word2vec_path(opt.w2v_dimension)
    embeddings, vocab_size, embeddings_dimension_size = load_embeddings_from_file(
        word2vec_path
    )

    # Siamese + Attention model
    model = SmashRNNModel(embeddings, vocab_size, embeddings_dimension_size, opt.level)

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


def get_args() -> argparse.Namespace:
    """Returns the arguments to run test steps

    Returns
    -------
    argparse.Namespace

    """
    parser = argparse.ArgumentParser(
        """Implementation of Siamese multi-attention RNN"""
    )
    parser.add_argument("--level", type=str, default="sentence")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--paragraphs_limit", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="base")
    parser.add_argument("--w2v_dimension", type=int, default=50)
    parser.add_argument("--introduction_only", type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    _opt = get_args()
    test(_opt)
