import argparse
import sys
import os

from tqdm import tqdm

from utils.constants import TEST_DATASET_PATH

os.chdir(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import logging

import pandas as pd
import torch
from torch import nn

from modeling.wikipedia2vec_model import Wikipedia2VecModel

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class TestWikipedia2Vec:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        self.batch_size = 32

        logger.info("Initializing parameters")
        click_stream_validation_dataset = torch.load(TEST_DATASET_PATH)
        validation_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "drop_last": False,
        }
        self.click_stream_test = torch.utils.data.DataLoader(
            click_stream_validation_dataset, **validation_params
        )

        self.wikipedia2vec = Wikipedia2VecModel()

        self.models = {"wikipedia2vec": self.wikipedia2vec}

        options = self.get_args()
        self.model_name = options.model_name

        self.criterion = nn.SmoothL1Loss()

        self.column_names = [
            "model",
            "source_article",
            "target_article",
            "actual_click_rate",
            "predicted_click_rate",
        ]

        logger.info("Parameters initialized")

    def run(self):
        for model_name, model in self.models.items():
            regression_model = torch.load(
                f"./trained_models/{str(model_name)}_{self.model_name}_regression_model"
            ).to(self.device)
            self.test(model_name, model, regression_model)

    def test(self, model_name, model, regression_model):
        loss_list = []
        predictions_list = pd.DataFrame(columns=self.column_names)

        logger.info(f"Model {model_name}. Starting evaluation")

        for row in tqdm(self.click_stream_test):
            source_article_vector = model.get_entity_vector(row["source_article"])
            target_article_vector = model.get_entity_vector(row["target_article"])

            siamese_representation = self.get_siamese_representation(
                source_article_vector, target_article_vector
            )

            prediction = regression_model(siamese_representation)

            loss = self.criterion(
                prediction.squeeze(1), row["click_rate"].to(self.device)
            )
            loss_list.append(loss)

            batch_results = pd.DataFrame(
                zip(
                    [model_name] * 32,
                    row["source_article"],
                    row["target_article"],
                    row["click_rate"].tolist(),
                    prediction.squeeze(1).tolist(),
                ),
                columns=self.column_names,
            )

            predictions_list = predictions_list.append(batch_results, ignore_index=True)

        final_loss = sum(loss_list) / len(loss_list)
        logger.info(
            f"Model {model_name}. Evaluation finished. Final loss: {final_loss}"
        )

        predictions_list.to_csv(
            f"./results/test/results_{model_name}_{self.model_name}_test.csv",
            index=False,
        )

    @staticmethod
    def get_siamese_representation(source_article, target_article):
        return torch.cat(
            (
                source_article,
                target_article,
                torch.abs(source_article - target_article),
                source_article * target_article,
            ),
            1,
        )

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            """Implementation of doc2vec to predict the number of clicks for Wikipedia articles"""
        )
        parser.add_argument("--model_name", type=str, default="base")

        return parser.parse_args()


if __name__ == "__main__":
    tester = TestWikipedia2Vec()
    tester.run()
