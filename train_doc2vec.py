import argparse
import logging
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.modeling.doc2vec_model import Doc2VecModel
from src.utils.constants import (
    ACTUAL_CLICK_RATE_COLUMN,
    CLICK_RATE_COLUMN,
    MODEL_COLUMN,
    PREDICTED_CLICK_RATE_COLUMN,
    SOURCE_ARTICLE_COLUMN,
    TARGET_ARTICLE_COLUMN,
    TRAIN_DATASET_PATH,
    VALIDATION_DATASET_PATH,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class TrainDoc2Vec:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        self.batch_size = 32

        click_stream_train_dataset = torch.load(TRAIN_DATASET_PATH)
        training_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "drop_last": True,
        }
        self.click_stream_train = torch.utils.data.DataLoader(
            click_stream_train_dataset, **training_params
        )

        click_stream_validation_dataset = torch.load(VALIDATION_DATASET_PATH)
        validation_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "drop_last": False,
        }
        self.click_stream_validation = torch.utils.data.DataLoader(
            click_stream_validation_dataset, **validation_params
        )

        self.doc2vec = Doc2VecModel()
        logger.info("Loaded models")

        self.models = {"doc2vec": self.doc2vec}

        options = self.get_args()

        self.num_epochs = options.num_epochs
        self.patience = options.patience
        self.model_name = options.model_name

        self.criterion = nn.SmoothL1Loss()

        self.column_names = [
            MODEL_COLUMN,
            SOURCE_ARTICLE_COLUMN,
            TARGET_ARTICLE_COLUMN,
            ACTUAL_CLICK_RATE_COLUMN,
            PREDICTED_CLICK_RATE_COLUMN,
        ]
        logger.info("Parameters initialized")

    @staticmethod
    def get_optimizer(regression_model):
        learning_rate = 10e-5

        return torch.optim.Adam(regression_model.parameters(), lr=learning_rate)

    def get_regression_model(self, hidden_size):
        input_dim = hidden_size * 4  # 3 = number of concatenations
        # Not mentioned in the paper.
        mlp_dim = int(input_dim / 2)
        output_dim = 1

        # nn.Sequential(
        #     nn.Linear(input_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, output_dim), nn.Sigmoid(),
        # ).to(self.device)

        return nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim),
        ).to(self.device)

    def train(self):
        for model_name, model in self.models.items():
            regression_model = self.get_regression_model(model.get_hidden_size())
            optimizer = self.get_optimizer(regression_model)

            num_epochs_without_improvement = 0
            best_loss = 1
            best_weights = None
            best_epoch = 0

            for epoch in range(self.num_epochs):
                regression_model.train()
                logger.info(
                    f"Model {model_name}. Starting epoch {epoch + 1} / {self.num_epochs}"
                )

                for row in tqdm(self.click_stream_train):
                    source_article_vector = model.get_entity_vector(
                        row[SOURCE_ARTICLE_COLUMN]
                    )
                    target_article_vector = model.get_entity_vector(
                        row[TARGET_ARTICLE_COLUMN]
                    )

                    optimizer.zero_grad()

                    siamese_representation = self.get_siamese_representation(
                        source_article_vector, target_article_vector
                    )

                    prediction = regression_model(siamese_representation)

                    loss = self.criterion(
                        prediction.squeeze(1).float(),
                        row[CLICK_RATE_COLUMN].to(self.device).float(),
                    )
                    loss.backward()
                    optimizer.step()

                logger.info(
                    f"Model {model_name}. Finished epoch {epoch + 1}/ {self.num_epochs}. Starting validation"
                )
                validation_loss = self.validation(model_name, model, regression_model)

                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_weights = {
                        k: v.to("cpu").clone()
                        for k, v in regression_model.state_dict().items()
                    }
                    best_epoch = epoch
                    num_epochs_without_improvement = 0
                else:
                    num_epochs_without_improvement += 1

                if num_epochs_without_improvement >= self.patience:
                    regression_model.load_state_dict(best_weights)
                    break

            logger.info(f"Training finished. Best epoch {best_epoch}")

            # This code may throw the warning UserWarning: Couldn't retrieve source code for container of type Sigmoid.
            # This warning is not problematic https://discuss.pytorch.org/t/got-warning-couldnt-retrieve-source-code-for-container/7689/13
            torch.save(
                regression_model,
                f"./trained_models/{str(model_name)}_{self.model_name}_regression_model",
            )

            logger.info("Models saved")

    def validation(self, model_name, model, regression_model):
        loss_list = []
        predictions_list = pd.DataFrame(columns=self.column_names)

        for row in tqdm(self.click_stream_validation):
            source_article_vector = model.get_inferred_vector(
                row[SOURCE_ARTICLE_COLUMN]
            )
            target_article_vector = model.get_inferred_vector(
                row[TARGET_ARTICLE_COLUMN]
            )

            with torch.no_grad():
                siamese_representation = self.get_siamese_representation(
                    source_article_vector, target_article_vector
                )

                prediction = regression_model(siamese_representation)

            loss = self.criterion(
                prediction.squeeze(1).float(),
                row[CLICK_RATE_COLUMN].to(self.device).float(),
            )
            loss_list.append(loss)

            batch_results = pd.DataFrame(
                zip(
                    [model_name] * 16,
                    row[SOURCE_ARTICLE_COLUMN],
                    row[TARGET_ARTICLE_COLUMN],
                    row[CLICK_RATE_COLUMN].tolist(),
                    prediction.squeeze(1).tolist(),
                ),
                columns=self.column_names,
            )

            predictions_list = predictions_list.append(batch_results, ignore_index=True)

        predictions_list.to_csv(
            "./results/results_{}_level_validation_{}.csv".format(
                model_name, datetime.now()
            ),
            index=False,
        )

        final_loss = sum(loss_list) / len(loss_list)
        logger.info(f"Model {model_name}. Validation loss: {final_loss}")

        return final_loss

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
        parser.add_argument("--num_epochs", type=int, default=30)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument("--model_name", type=str, default="base")

        return parser.parse_args()


if __name__ == "__main__":
    train = TrainDoc2Vec()
    train.train()
