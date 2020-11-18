"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""

import sys
import os
from typing import List, Union

os.chdir(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

from utils.constants import RESULT_FILE_COLUMNS_NAMES, WIKI_ARTICLES_DATASET_PATH

from tqdm import tqdm
import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
from modeling.smash_rnn_model import SmashRNNModel
from modeling.smash_dataset import SMASHDataset
from datetime import datetime
from utils.utils import get_word2vec_path, load_embeddings_from_file

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"
TEXT_IDS_COLUMN = "text_ids"
CLICK_RATE_COLUMN = "click_rate"
TARGET_ARTICLE_COLUMN = "target_article"
SOURCE_ARTICLE_COLUMN = "source_article"

TRAIN_DATASET_PATH = "./data/dataset/click_stream_train.pth"
VALIDATION_DATASET_PATH = "./data/dataset/click_stream_validation.pth"
TEST_DATASET_PATH = "./data/dataset/click_stream_test.pth"
MODEL_FOLDER = "./trained_models/"
RESULTS_PATH = "./results/validation/"


class SmashRNN:
    """
    Class to run the SmashRNN training steps

    """

    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        # Basic config. Should be customizable in the future
        self.learning_rate = 10e-5
        self.patience = 5

        self.opt = self.get_args()
        self.num_validations = int(self.opt.num_epochs / self.opt.validation_interval)
        # End of configs

        self.model_name = (
            self.opt.model_name + "_introduction_only"
            if self.opt.introduction_only
            else self.opt.model_name
        )

        self.articles = SMASHDataset(
            WIKI_ARTICLES_DATASET_PATH, introduction_only=self.opt.introduction_only
        )

        word2vec_path = get_word2vec_path(self.opt.w2v_dimension)

        # Load from txt file (in word2vec format)
        embeddings, vocab_size, embeddings_dimension_size = load_embeddings_from_file(
            word2vec_path
        )

        self.model = SmashRNNModel(
            embeddings, vocab_size, embeddings_dimension_size, levels=self.opt.level
        )
        self.model.to(self.device)

        # Overall model optimization and evaluation parameters
        self.criterion = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.batch_size = self.opt.batch_size

        self.model.train()

    def train(self, level: str = "paragraph") -> None:
        """Executes the training steps of SMASH RNN

        Parameters
        ----------
        level : str
            The deepest hierarchy level to calculate the model

        Returns
        -------
        None
        """
        click_stream_train = torch.load(TRAIN_DATASET_PATH)
        training_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "drop_last": True,
        }
        training_generator = torch.utils.data.DataLoader(
            click_stream_train, **training_params
        )

        paragraphs_limit = (
            self.opt.paragraphs_limit
            if self.opt.paragraphs_limit is not None
            else self.articles.get_n_percentile_paragraph_length()
        )

        print("Starting training {}".format(datetime.now()))

        num_epochs_without_improvement = 0
        best_loss = 1
        best_weights = None
        best_epoch = 0

        print(range(self.opt.num_epochs))

        for epoch in range(self.opt.num_epochs):
            self.model.train()

            loss_list = []

            for row in tqdm(training_generator):
                source_articles = self.articles.get_articles(row[SOURCE_ARTICLE_COLUMN])
                target_articles = self.articles.get_articles(row[TARGET_ARTICLE_COLUMN])

                row[CLICK_RATE_COLUMN] = row[CLICK_RATE_COLUMN].to(self.device)

                self.optimizer.zero_grad()

                predictions = self.model(
                    target_articles,
                    source_articles,
                    paragraphs_limit,
                )
                del source_articles
                del target_articles

                loss = self.criterion(
                    predictions.squeeze(1).float(), row[CLICK_RATE_COLUMN].float()
                )
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())

            loss = self.calculate_loss(loss_list)

            print(
                "Epoch: {}/{}, Lr: {}, Loss: {}, Time: {}".format(
                    epoch + 1,
                    self.opt.num_epochs,
                    self.optimizer.param_groups[0]["lr"],
                    loss,
                    datetime.now(),
                )
            )

            validation_loss = self.validate(epoch, level)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = {
                    k: v.to("cpu").clone() for k, v in self.model.state_dict().items()
                }
                best_epoch = epoch
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1

            if num_epochs_without_improvement >= self.patience:
                self.model.load_state_dict(best_weights)
                break

        self.save_model()
        print(f"Training finished {datetime.now()}. Best epoch {best_epoch + 1}")

    def validate(self, validation_step: int, level: str) -> Union[float, torch.Tensor]:
        """Executes a validation step of the model at the defined level. Returns
        the validation loss for the step.

        Parameters
        ----------
        validation_step : int
            The current validation step
        level : str
            The deepest level to calculate the model

        Returns
        -------
        Union[float, torch.Tensor]

        """
        click_stream_validation = torch.load(VALIDATION_DATASET_PATH)

        validation_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "drop_last": False,
        }
        validation_generator = torch.utils.data.DataLoader(
            click_stream_validation, **validation_params
        )
        validation_step = int(validation_step) + 1

        loss_list = []

        predictions_list = pd.DataFrame(columns=RESULT_FILE_COLUMNS_NAMES)

        for row in tqdm(validation_generator):
            source_articles = self.articles.get_articles(row[SOURCE_ARTICLE_COLUMN])
            target_articles = self.articles.get_articles(row[TARGET_ARTICLE_COLUMN])

            row[CLICK_RATE_COLUMN] = row[CLICK_RATE_COLUMN].to(self.device)

            with torch.no_grad():
                predictions = self.model(target_articles, source_articles)

            loss = self.criterion(predictions.squeeze(1), row[CLICK_RATE_COLUMN])

            loss_list.append(loss)

            batch_results = pd.DataFrame(
                zip(
                    [self.model_name] * self.batch_size,
                    row[SOURCE_ARTICLE_COLUMN],
                    row[TARGET_ARTICLE_COLUMN],
                    row[CLICK_RATE_COLUMN].tolist(),
                    predictions.squeeze(1).tolist(),
                ),
                columns=RESULT_FILE_COLUMNS_NAMES,
            )

            predictions_list = predictions_list.append(batch_results, ignore_index=True)

        final_loss = self.calculate_loss(loss_list)

        predictions_list.to_csv(
            "{}results_{}_level_validation_{}.csv".format(
                RESULTS_PATH, self.model_name, datetime.now()
            ),
            index=False,
        )

        print(
            "{} level\n Validation: {}/{}, Lr: {}, Loss: {}".format(
                level.capitalize(),
                validation_step,
                self.num_validations,
                self.optimizer.param_groups[0]["lr"],
                final_loss,
            )
        )

        return round(final_loss.item(), 8)

    def save_model(self) -> None:
        """Saves the current model in the appropriate folder

        Returns
        -------
        None

        """
        model_path = f"{MODEL_FOLDER}{self.opt.level}_level_{self.model_name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

    @staticmethod
    def calculate_loss(
        loss_list: List[Union[float, torch.Tensor]]
    ) -> Union[float, torch.Tensor]:
        """Calculates the loss for a specific epoch

        Parameters
        ----------
        loss_list : List[Union[float, torch.Tensor]]
            List with all losses from a specific epoch

        Returns
        -------
        Union[float, torch.Tensor]

        """
        return sum(loss_list) / len(loss_list)

    @staticmethod
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            """Implementation of the model described in the paper: Semantic Text Matching for Long-Form Documents to predict the number of clicks for Wikipedia articles"""
        )

        parser.add_argument("--num_epochs", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--level", type=str, default="paragraph")
        parser.add_argument("--paragraphs_limit", type=int, default=None)
        parser.add_argument("--model_name", type=str, default="base")
        parser.add_argument("--w2v_dimension", type=int, default=50)
        parser.add_argument("--introduction_only", type=bool, default=False)

        return parser.parse_args()

    def run(self):
        model.train(self.opt.level)


if __name__ == "__main__":
    model = SmashRNN()
    model.run()
