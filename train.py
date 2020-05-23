"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""

import sys
import os

from tqdm import tqdm

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import argparse
import csv
import logging
import numpy as np
import pandas as pd
from comet_ml import Experiment
import torch
import torch.nn as nn
from utils.utils import (
    get_document_at_sentence_level,
    get_words_per_sentence_at_sentence_level,
    get_document_at_word_level,
    get_sentences_per_paragraph_at_sentence_level,
    get_words_per_document_at_word_level,
)
from modeling.smash_rnn_model import SmashRNNModel
from modeling.smash_dataset import SMASHDataset
from datetime import datetime

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"
TEXT_IDS_COLUMN = "text_ids"
CLICK_RATE_COLUMN = "click_rate"
TARGET_ARTICLE_COLUMN = "target_article"
SOURCE_ARTICLE_COLUMN = "source_article"

if torch.cuda.is_available():
    WIKI_ARTICLES_DATASET_PATH = "~/thesis-davi/data/dataset/wiki_articles_english.csv"
else:
    WIKI_ARTICLES_DATASET_PATH = "./data/dataset/wiki_articles_english.csv"


class SmashRNN:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        # Basic config. Should be customizable in the future
        self.learning_rate = 10e-5
        self.patience = 3

        self.opt = self.get_args()
        self.batch_size = self.opt.batch_size
        self.num_validations = int(self.opt.num_epoches / self.opt.validation_interval)
        # End of configs

        self.articles = SMASHDataset(WIKI_ARTICLES_DATASET_PATH)

        # Load from txt file (in word2vec format)
        dict = pd.read_csv(
            filepath_or_buffer=self.opt.word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
        ).values[:, 1:]
        dict_len, embed_dim = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_dim))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        # Paragraph level model
        self.model = SmashRNNModel(dict, dict_len, embed_dim)
        self.model.to(self.device)

        # Overall model optimization and evaluation parameters
        self.criterion = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()

        if torch.cuda.is_available():
            self.experiment = Experiment(
                api_key="NPD7aHoJxhZgG0MNWBkFb3hzZ", project_name="thesis-davi", workspace="davipaula",
            )

    def train(self, level="paragraph"):
        training_generator = torch.load(self.opt.train_dataset_path)
        print("Starting training {}".format(datetime.now()))

        num_epochs_without_improvement = 0
        best_loss = 1
        best_weights = None
        best_epoch = 0

        for epoch in range(self.opt.num_epoches):
            self.model.train()

            loss_list = []

            for row in tqdm(training_generator):
                source_articles = self.articles.get_articles(row[SOURCE_ARTICLE_COLUMN])
                target_articles = self.articles.get_articles(row[TARGET_ARTICLE_COLUMN])

                if level == "sentence":
                    source_articles = self.transform_to_sentence_level(source_articles)
                    target_articles = self.transform_to_sentence_level(target_articles)

                elif level == "word":
                    source_articles = self.transform_to_word_level(source_articles)
                    target_articles = self.transform_to_word_level(target_articles)

                if torch.cuda.is_available():
                    row[CLICK_RATE_COLUMN] = row[CLICK_RATE_COLUMN].cuda()
                    source_articles[TEXT_IDS_COLUMN] = source_articles[TEXT_IDS_COLUMN].cuda()
                    source_articles[WORDS_PER_SENTENCE_COLUMN] = source_articles[WORDS_PER_SENTENCE_COLUMN].cuda()
                    source_articles[SENTENCES_PER_PARAGRAPH_COLUMN] = source_articles[
                        SENTENCES_PER_PARAGRAPH_COLUMN
                    ].cuda()
                    source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN] = source_articles[
                        PARAGRAPHS_PER_DOCUMENT_COLUMN
                    ].cuda()
                    target_articles[TEXT_IDS_COLUMN] = source_articles[TEXT_IDS_COLUMN].cuda()
                    target_articles[WORDS_PER_SENTENCE_COLUMN] = source_articles[WORDS_PER_SENTENCE_COLUMN].cuda()
                    target_articles[SENTENCES_PER_PARAGRAPH_COLUMN] = source_articles[
                        SENTENCES_PER_PARAGRAPH_COLUMN
                    ].cuda()
                    target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN] = source_articles[
                        PARAGRAPHS_PER_DOCUMENT_COLUMN
                    ].cuda()

                self.optimizer.zero_grad()

                predictions = self.model(
                    target_articles[TEXT_IDS_COLUMN],
                    target_articles[WORDS_PER_SENTENCE_COLUMN],
                    target_articles[SENTENCES_PER_PARAGRAPH_COLUMN],
                    target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN],
                    source_articles[TEXT_IDS_COLUMN],
                    source_articles[WORDS_PER_SENTENCE_COLUMN],
                    source_articles[SENTENCES_PER_PARAGRAPH_COLUMN],
                    source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN],
                )

                loss = self.criterion(predictions.squeeze(1), row[CLICK_RATE_COLUMN])
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss)

            loss = self.calculate_loss(loss_list)

            if torch.cuda.is_available():
                self.experiment.log_metric("train_loss", loss.item(), epoch=epoch + 1)

            print(
                "Epoch: {}/{}, Lr: {}, Loss: {}, Time: {}".format(
                    epoch + 1, self.opt.num_epoches, self.optimizer.param_groups[0]["lr"], loss, datetime.now(),
                )
            )

            validation_loss = self.validate(int(epoch / self.opt.validation_interval), level)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = {k: v.to("cpu").clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1

            if num_epochs_without_improvement >= self.patience:
                self.model.load_state_dict(best_weights)
                break

        self.save_model()
        print("Training finished {}. Best epoch {}".format(datetime.now(), best_epoch))

    @staticmethod
    def transform_to_word_level(document):
        batch_size = document[TEXT_IDS_COLUMN].shape[0]

        document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_document_at_word_level(document[WORDS_PER_SENTENCE_COLUMN])
        document[TEXT_IDS_COLUMN] = get_document_at_word_level(
            document[TEXT_IDS_COLUMN], document[WORDS_PER_SENTENCE_COLUMN]
        )
        document[SENTENCES_PER_PARAGRAPH_COLUMN] = torch.ones((batch_size, 1), dtype=int)
        document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(batch_size, dtype=int)

        return document

    @staticmethod
    def transform_to_sentence_level(document):
        batch_size = document[TEXT_IDS_COLUMN].shape[0]

        document[TEXT_IDS_COLUMN] = get_document_at_sentence_level(document[TEXT_IDS_COLUMN])
        document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_sentence_at_sentence_level(
            document[WORDS_PER_SENTENCE_COLUMN]
        )
        document[SENTENCES_PER_PARAGRAPH_COLUMN] = get_sentences_per_paragraph_at_sentence_level(
            document[SENTENCES_PER_PARAGRAPH_COLUMN]
        )
        document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(batch_size, dtype=int)

        return document

    def validate(self, validation_step, level):
        validation_generator = torch.load(self.opt.validation_dataset_path)
        validation_step = int(validation_step) + 1

        loss_list = []
        columns_names = [
            "source_document",
            "target_document",
            "actual_click_rate",
            "predicted_click_rate",
        ]
        predictions_list = pd.DataFrame(columns=columns_names)

        for row in validation_generator:
            source_articles = self.articles.get_articles(row[SOURCE_ARTICLE_COLUMN])
            target_articles = self.articles.get_articles(row[TARGET_ARTICLE_COLUMN])

            if level == "sentence":
                source_articles = self.transform_to_sentence_level(source_articles)
                target_articles = self.transform_to_sentence_level(target_articles)

            elif level == "word":
                source_articles = self.transform_to_word_level(source_articles)
                target_articles = self.transform_to_word_level(target_articles)

            if torch.cuda.is_available():
                row[CLICK_RATE_COLUMN] = row[CLICK_RATE_COLUMN].cuda()
                source_articles[TEXT_IDS_COLUMN] = source_articles[TEXT_IDS_COLUMN].cuda()
                source_articles[WORDS_PER_SENTENCE_COLUMN] = source_articles[WORDS_PER_SENTENCE_COLUMN].cuda()
                source_articles[SENTENCES_PER_PARAGRAPH_COLUMN] = source_articles[SENTENCES_PER_PARAGRAPH_COLUMN].cuda()
                source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN] = source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN].cuda()
                target_articles[TEXT_IDS_COLUMN] = target_articles[TEXT_IDS_COLUMN].cuda()
                target_articles[WORDS_PER_SENTENCE_COLUMN] = target_articles[WORDS_PER_SENTENCE_COLUMN].cuda()
                target_articles[SENTENCES_PER_PARAGRAPH_COLUMN] = target_articles[SENTENCES_PER_PARAGRAPH_COLUMN].cuda()
                target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN] = target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN].cuda()

            with torch.no_grad():
                predictions = self.model(
                    target_articles[TEXT_IDS_COLUMN],
                    target_articles[WORDS_PER_SENTENCE_COLUMN],
                    target_articles[SENTENCES_PER_PARAGRAPH_COLUMN],
                    target_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN],
                    source_articles[TEXT_IDS_COLUMN],
                    source_articles[WORDS_PER_SENTENCE_COLUMN],
                    source_articles[SENTENCES_PER_PARAGRAPH_COLUMN],
                    source_articles[PARAGRAPHS_PER_DOCUMENT_COLUMN],
                )

            loss = self.criterion(predictions.squeeze(1), row[CLICK_RATE_COLUMN])

            loss_list.append(loss)

            batch_results = pd.DataFrame(
                zip(
                    row[SOURCE_ARTICLE_COLUMN],
                    row[TARGET_ARTICLE_COLUMN],
                    row[CLICK_RATE_COLUMN].tolist(),
                    predictions.squeeze(1).tolist(),
                ),
                columns=columns_names,
            )

            predictions_list = predictions_list.append(batch_results, ignore_index=True)

        final_loss = self.calculate_loss(loss_list)

        if torch.cuda.is_available():
            self.experiment.log_metric("validation_loss", final_loss.item(), epoch=validation_step)

        predictions_list.to_csv(
            "{}results_{}_level_validation_{}.csv".format(self.opt.results_path, level, datetime.now()), index=False,
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

    def save_model(self):
        model_path = self.opt.model_folder + self.opt.level + "_level_model.pt"
        torch.save(self.model.state_dict(), model_path)

    def should_stop(self, epoch):
        return epoch - self.best_epoch > self.es_patience > 0

    def should_run_validation(self, epoch):
        return ((epoch + 1) % self.opt.validation_interval) == 0

    @staticmethod
    def calculate_loss(loss_list):
        return sum(loss_list) / len(loss_list)

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            """Implementation of the model described in the paper: Semantic Text Matching for Long-Form Documents to predict the number of clicks for Wikipedia articles"""
        )
        parser.add_argument("--model_folder", type=str, default="./trained_models/")
        parser.add_argument("--full_dataset_path", type=str, default="./data/dataset/click_stream_train.pth")
        parser.add_argument("--word2vec_path", type=str, default="./data/glove.6B.50d.txt")
        parser.add_argument("--train_dataset_path", type=str, default="./data/dataset/click_stream_train.pth")
        parser.add_argument("--validation_dataset_path", type=str, default="./data/dataset/click_stream_validation.pth")
        parser.add_argument("--test_dataset_path", type=str, default="./data/dataset/click_stream_test.pth")
        parser.add_argument("--num_epoches", type=int, default=1)
        parser.add_argument("--validation_interval", type=int, default=1)
        parser.add_argument(
            "--limit_rows_dataset",
            type=int,
            default=9999999,
            help="For development purposes. This limits the number of rows read from the dataset.",
        )
        parser.add_argument("--level", type=str, default="paragraph")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--results_path", type=str, default="./")

        return parser.parse_args()

    def run(self):
        model.train(self.opt.level)


if __name__ == "__main__":
    model = SmashRNN()
    model.run()
