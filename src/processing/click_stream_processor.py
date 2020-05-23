import numpy as np
import pandas as pd
import os
import random
import torch
from datetime import datetime
from data_structure.click_stream_dataset import ClickStreamDataset
from data_structure.click_stream_pre_processed import ClickStreamPreProcessed
from torch.utils.data import TensorDataset, random_split

NUMBER_OF_CLICKS_COLUMN = "number_of_clicks"

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
        self.dataset = self.filter_dataset(
            _pre_processed_dataset, self.get_available_titles_in_wiki_articles()
        )

    @staticmethod
    def get_available_titles_in_wiki_articles():
        # with open(
        #     "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/available_titles.txt"
        # ) as f:
        #     selected_articles = f.read().splitlines()

        df_wiki = pd.read_csv(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/wiki_articles_english.csv"
        )
        selected_articles = df_wiki["article"].tolist()

        return selected_articles

    def filter_dataset(self, click_stream, available_titles, threshold: int = 10000):
        click_stream = self.filter_available_titles(click_stream, available_titles)
        click_stream = self.filter_number_of_clicks_treshold(click_stream, threshold)

        return click_stream

    @staticmethod
    def filter_available_titles(click_stream, available_titles):
        filtered_dataset = click_stream[
            (click_stream[SOURCE_ARTICLE_COLUMN].isin(available_titles))
            & (click_stream[TARGET_ARTICLE_COLUMN].isin(available_titles))
        ].copy()

        return filtered_dataset

    @staticmethod
    def filter_number_of_clicks_treshold(click_stream, total_number_of_clicks):
        number_of_clicks_per_source_article = click_stream.groupby(
            SOURCE_ARTICLE_COLUMN
        )[NUMBER_OF_CLICKS_COLUMN].sum()
        articles_above_threshold = number_of_clicks_per_source_article[
            number_of_clicks_per_source_article > total_number_of_clicks
        ].index

        return click_stream[
            (click_stream[SOURCE_ARTICLE_COLUMN].isin(articles_above_threshold))
            & (click_stream[NUMBER_OF_CLICKS_COLUMN] > 200)
        ]

    def run(
        self,
        train_split=0.8,
        batch_size=32,
        save_folder="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/",
    ):
        print("Beginning of dataset split")

        train_dataset, validation_dataset, test_dataset = self.split_datasets()

        click_stream_train = ClickStreamDataset(train_dataset)
        click_stream_validation = ClickStreamDataset(validation_dataset)
        click_stream_test = ClickStreamDataset(test_dataset)

        print("Datasets split. Starting saving them", datetime.now())

        training_params = {"batch_size": batch_size, "shuffle": True, "drop_last": True}
        train_loader = torch.utils.data.DataLoader(
            click_stream_train, **training_params
        )

        validation_and_test_params = {
            "batch_size": batch_size,
            "shuffle": True,
            "drop_last": False,
        }
        validation_loader = torch.utils.data.DataLoader(
            click_stream_validation, **validation_and_test_params
        )

        test_loader = torch.utils.data.DataLoader(
            click_stream_test, **validation_and_test_params
        )

        torch.save(
            train_loader, os.path.join(save_folder, "dataset", "click_stream_train.pth")
        )
        torch.save(
            validation_loader,
            os.path.join(save_folder, "dataset", "click_stream_validation.pth"),
        )
        torch.save(
            test_loader, os.path.join(save_folder, "dataset", "click_stream_test.pth")
        )

        print("Datasets saved successfully")

    def generate_dataset_sample(
        self,
        sample_size=1,
        destination_path="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/click_stream_random_sample.csv",
    ):
        random.seed(123)

        unique_source_articles = (
            self.dataset[SOURCE_ARTICLE_COLUMN].drop_duplicates().reset_index(drop=True)
        )

        num_articles = len(unique_source_articles)
        sample_dataset_size = int(num_articles * sample_size)

        selected_indices = random.sample(range(num_articles), sample_dataset_size)

        selected_articles = unique_source_articles[
            unique_source_articles.index.isin(selected_indices)
        ]

        dataset_sample = self.dataset[
            self.dataset[SOURCE_ARTICLE_COLUMN].isin(selected_articles)
        ].reset_index(drop=True)
        dataset_sample.to_csv(destination_path, index=False)

        self.save_selected_articles_file(dataset_sample)

        return dataset_sample

    @staticmethod
    def save_selected_articles_file(dataset):
        selected_articles = pd.Series.append(
            dataset[SOURCE_ARTICLE_COLUMN].drop_duplicates(),
            dataset[TARGET_ARTICLE_COLUMN].drop_duplicates(),
        ).reset_index(drop=True)

        with open(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/selected_articles.txt",
            "w",
        ) as output:
            output.write("\n".join(selected_articles))

    def split_datasets(self, train_split=0.8):
        random.seed(123)
        click_stream_data = self.generate_dataset_sample()

        source_articles = click_stream_data[SOURCE_ARTICLE_COLUMN].unique().tolist()

        click_stream_size = len(source_articles)
        train_dataset_size = int(click_stream_size * train_split)
        validation_dataset_size = int((click_stream_size - train_dataset_size) / 2)
        test_dataset_size = (
            click_stream_size - train_dataset_size - validation_dataset_size
        )

        random.shuffle(source_articles)

        train_articles = source_articles[:train_dataset_size]
        validation_articles = source_articles[
            train_dataset_size : (train_dataset_size + validation_dataset_size)
        ]
        test_articles = source_articles[-test_dataset_size:]

        train_validation = set(train_articles + validation_articles)
        train_test = set(train_articles + test_articles)

        train_dataset = click_stream_data[
            click_stream_data[SOURCE_ARTICLE_COLUMN].isin(train_articles)
            & click_stream_data[TARGET_ARTICLE_COLUMN].isin(train_articles)
        ]

        validation_dataset = click_stream_data[
            click_stream_data[SOURCE_ARTICLE_COLUMN].isin(validation_articles)
            & ~click_stream_data[TARGET_ARTICLE_COLUMN].isin(train_test)
        ]

        test_dataset = click_stream_data[
            click_stream_data[SOURCE_ARTICLE_COLUMN].isin(test_articles)
            & ~click_stream_data[TARGET_ARTICLE_COLUMN].isin(train_validation)
        ]

        train_articles_dict = [
            (article, "train")
            for article in set(
                train_dataset[SOURCE_ARTICLE_COLUMN].tolist()
                + train_dataset[TARGET_ARTICLE_COLUMN].tolist()
            )
        ]

        validation_articles_dict = [
            (article, "validation")
            for article in set(
                validation_dataset[SOURCE_ARTICLE_COLUMN].tolist()
                + validation_dataset[TARGET_ARTICLE_COLUMN].tolist()
            )
        ]

        test_articles_dict = [
            (article, "test")
            for article in set(
                test_dataset[SOURCE_ARTICLE_COLUMN].tolist()
                + test_dataset[TARGET_ARTICLE_COLUMN].tolist()
            )
        ]

        articles_fold = pd.DataFrame(
            train_articles_dict + validation_articles_dict + test_articles_dict,
            columns=["article", "fold"],
        )

        articles_fold.to_csv(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/selected_articles.csv",
            index=False,
        )

        return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    creator = ClickStreamProcessor()

    creator.run()
