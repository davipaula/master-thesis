import numpy as np
import pandas as pd
import os
import random
import torch
from datetime import datetime
from data_structure.click_stream_dataset import ClickStreamDataset
from data_structure.click_stream_pre_processed import ClickStreamPreProcessed
from torch.utils.data import TensorDataset, random_split

SOURCE_ARTICLE_COLUMN = "source_article"
TARGET_ARTICLE_COLUMN = "target_article"


class ClickStreamProcessor:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        self._click_stream_pre_processed = ClickStreamPreProcessed()

        _pre_processed_dataset = ClickStreamPreProcessed().dataset
        self.dataset = self.filter_click_stream_data(
            _pre_processed_dataset, self.get_available_titles_in_wiki_articles()
        )

    @staticmethod
    def get_available_titles_in_wiki_articles():
        with open(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/available_titles.txt"
        ) as f:
            selected_articles = f.read().splitlines()

        return selected_articles

    @staticmethod
    def filter_click_stream_data(click_stream, available_titles):
        filtered_dataset = click_stream[
            (click_stream[SOURCE_ARTICLE_COLUMN].isin(available_titles))
            & (click_stream[TARGET_ARTICLE_COLUMN].isin(available_titles))
        ].copy()

        return filtered_dataset

    def run(
        self,
        train_split=0.8,
        batch_size=32,
        save_folder="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/",
    ):
        print("Beginning of dataset split")

        click_stream_data = self.generate_dataset_sample()

        click_stream_size = len(click_stream_data)
        train_dataset_size = int(click_stream_size * train_split)
        validation_dataset_size = int((click_stream_size - train_dataset_size) / 2)
        test_dataset_size = click_stream_size - train_dataset_size - validation_dataset_size

        train_dataset, validation_dataset, test_dataset = random_split(
            click_stream_data, [train_dataset_size, validation_dataset_size, test_dataset_size],
        )

        selected_train_dataset = train_dataset.dataset.iloc[train_dataset.indices]
        selected_validation_dataset = validation_dataset.dataset.iloc[validation_dataset.indices]
        selected_test_dataset = test_dataset.dataset.iloc[test_dataset.indices]

        # This block removes from validation set articles that are in test dataset and from train set articles that are
        # in validation or test sets
        test_articles = set(
            selected_test_dataset[SOURCE_ARTICLE_COLUMN].to_list()
            + selected_test_dataset[TARGET_ARTICLE_COLUMN].to_list()
        )

        validation_articles = set(
            selected_validation_dataset[SOURCE_ARTICLE_COLUMN].to_list()
            + selected_validation_dataset[TARGET_ARTICLE_COLUMN].to_list()
        ).difference(test_articles)

        train_articles = (
            set(
                selected_train_dataset[SOURCE_ARTICLE_COLUMN].to_list()
                + selected_train_dataset[TARGET_ARTICLE_COLUMN].to_list()
            )
            .difference(validation_articles)
            .difference(test_articles)
        )

        train_dataset = ClickStreamDataset(
            train_dataset.dataset[
                train_dataset.dataset[SOURCE_ARTICLE_COLUMN].isin(train_articles)
                & train_dataset.dataset[TARGET_ARTICLE_COLUMN].isin(train_articles)
            ]
        )

        validation_dataset = ClickStreamDataset(
            validation_dataset.dataset[
                validation_dataset.dataset[SOURCE_ARTICLE_COLUMN].isin(train_articles)
                & validation_dataset.dataset[TARGET_ARTICLE_COLUMN].isin(train_articles)
            ]
        )

        test_dataset = ClickStreamDataset(
            validation_dataset.dataset[
                validation_dataset.dataset[SOURCE_ARTICLE_COLUMN].isin(train_articles)
                & validation_dataset.dataset[TARGET_ARTICLE_COLUMN].isin(train_articles)
            ]
        )

        train_articles_dict = [(article, "train") for article in train_articles]
        validation_articles_dict = [(article, "validation") for article in validation_articles]
        test_articles_dict = [(article, "test") for article in test_articles]

        articles_fold = pd.DataFrame(
            train_articles_dict + validation_articles_dict + test_articles_dict, columns=["article", "fold"]
        )

        articles_fold.to_csv(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/selected_articles.csv",
            index=False
        )

        print("Datasets split. Starting saving them", datetime.now())

        training_params = {"batch_size": batch_size, "shuffle": True, "drop_last": True}
        train_loader = torch.utils.data.DataLoader(train_dataset, **training_params)

        validation_and_test_params = {
            "batch_size": batch_size,
            "shuffle": True,
            "drop_last": False,
        }
        validation_loader = torch.utils.data.DataLoader(validation_dataset, **validation_and_test_params)

        test_loader = torch.utils.data.DataLoader(test_dataset, **validation_and_test_params)

        torch.save(train_loader, os.path.join(save_folder, "dataset", "click_stream_train.pth"))
        torch.save(
            validation_loader, os.path.join(save_folder, "dataset", "click_stream_validation.pth"),
        )
        torch.save(test_loader, os.path.join(save_folder, "dataset", "click_stream_test.pth"))

        print("Datasets saved successfully")

    def generate_dataset_sample(
        self,
        sample_size=0.01,
        destination_path="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/click_stream_random_sample.csv",
    ):
        random.seed(123)

        unique_source_articles = self.dataset[SOURCE_ARTICLE_COLUMN].drop_duplicates().reset_index(drop=True)

        num_articles = len(unique_source_articles)
        sample_dataset_size = int(num_articles * sample_size)

        selected_indices = random.sample(range(num_articles), sample_dataset_size)

        selected_articles = unique_source_articles[unique_source_articles.index.isin(selected_indices)]

        dataset_sample = self.dataset[self.dataset[SOURCE_ARTICLE_COLUMN].isin(selected_articles)].reset_index(
            drop=True
        )
        dataset_sample.to_csv(destination_path, index=False)

        self.save_selected_articles_file(dataset_sample)

        return dataset_sample

    @staticmethod
    def save_selected_articles_file(dataset):
        selected_articles = pd.Series.append(dataset[SOURCE_ARTICLE_COLUMN], dataset[TARGET_ARTICLE_COLUMN])

        with open(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/selected_articles.txt", "w"
        ) as output:
            output.write("\n".join(selected_articles))


if __name__ == "__main__":
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    creator = ClickStreamProcessor()

    creator.run()
