import os
import random
from ast import literal_eval
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split

from click_stream_dataset import ClickStreamDataset
from wiki_articles_dataset import WikiArticlesDataset
from click_stream_pre_processed import ClickStreamPreProcessed


class DatasetCreation:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        self.wiki_articles_pre_processed = pd.read_csv(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/wiki_articles.csv"
        )

        self.click_stream_pre_processed = ClickStreamPreProcessed(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/click_stream.csv"
        )

        self.dataset = self.generate_dataset()

    def generate_dataset(self):
        click_stream = self.get_click_stream_dataset_with_negative_sampling()
        click_stream = self.filter_click_stream_data(
            click_stream, self.get_available_titles_in_wiki_articles()
        )

        return click_stream

    def get_available_titles_in_wiki_articles(self):
        return self.wiki_articles_pre_processed["article"].unique()

    @staticmethod
    def filter_click_stream_data(click_stream, available_titles):
        filtered_dataset = click_stream[
            (click_stream["source_article"].isin(available_titles))
            & (click_stream["target_article"].isin(available_titles))
        ].copy()

        return filtered_dataset

    def get_click_stream_dataset_with_negative_sampling(self):
        click_stream_pre_processed_dataset = self.click_stream_pre_processed.dataset

        # Removing bad data. Should be fixed in the json parser
        wiki_dataset = self.wiki_articles_pre_processed[
            self.wiki_articles_pre_processed["text_ids"].str.find("#Redirect") == -1
        ].reset_index(drop=True)

        source_articles = (
            pd.merge(
                click_stream_pre_processed_dataset[["source_article"]],
                wiki_dataset[["article", "links"]],
                left_on=["source_article"],
                right_on=["article"],
            )
            .drop(["article"], axis=1)
            .drop_duplicates(subset="source_article")
        )

        source_articles["links"] = source_articles["links"].map(literal_eval)
        source_articles = source_articles.explode("links")
        visited_articles = click_stream_pre_processed_dataset[
            ["source_article", "target_article"]
        ]

        non_visited_articles = pd.merge(
            source_articles,
            visited_articles,
            left_on=["source_article", "links"],
            right_on=["source_article", "target_article"],
            how="left",
        )
        non_visited_articles = non_visited_articles[
            ~non_visited_articles["target_article"].isna()
        ]

        # Adds the non visited links data
        negative_sampling = non_visited_articles.drop(["target_article"], axis=1)
        negative_sampling.columns = ["source_article", "target_article"]

        negative_sampling.insert(len(negative_sampling.columns), "number_of_clicks", 0)
        negative_sampling.insert(len(negative_sampling.columns), "click_rate", 0)

        combined_dataset = click_stream_pre_processed_dataset.append(
            negative_sampling, ignore_index=True
        )

        return combined_dataset

    def split_dataset(
        self,
        train_split=0.8,
        batch_size=32,
        save_folder="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/",
    ):
        print("Beginning of dataset split")

        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        click_stream_data = self.generate_dataset_sample()

        click_stream_size = len(click_stream_data)
        train_dataset_size = int(click_stream_size * train_split)
        validation_dataset_size = int((click_stream_size - train_dataset_size) / 2)
        test_dataset_size = (
            click_stream_size - train_dataset_size - validation_dataset_size
        )

        train_dataset, validation_dataset, test_dataset = random_split(
            click_stream_data,
            [train_dataset_size, validation_dataset_size, test_dataset_size],
        )

        train_articles = np.unique(
            train_dataset.dataset[["source_article", "target_article"]]
            .iloc[train_dataset.indices]
            .astype(str)
        )
        train_wiki = self.wiki_articles_pre_processed[
            self.wiki_articles_pre_processed["article"].isin(train_articles)
        ]

        validation_articles = np.unique(
            validation_dataset.dataset[["source_article", "target_article"]]
            .iloc[validation_dataset.indices]
            .astype(str)
        )
        validation_wiki = self.wiki_articles_pre_processed[
            self.wiki_articles_pre_processed["article"].isin(validation_articles)
        ]

        test_articles = np.unique(
            validation_dataset.dataset[["source_article", "target_article"]]
            .iloc[test_dataset.indices]
            .astype(str)
        )
        test_wiki = self.wiki_articles_pre_processed[
            self.wiki_articles_pre_processed["article"].isin(test_articles)
        ]

        train_wiki["fold"] = "train"
        validation_wiki["fold"] = "validation"
        test_wiki["fold"] = "test"

        wiki_dataset = pd.concat([train_wiki, validation_wiki, test_wiki])

        print("Datasets split. Starting saving them", datetime.now())

        training_params = {"batch_size": batch_size, "shuffle": True, "drop_last": True}
        train_dataset = ClickStreamDataset(train_dataset.dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, **training_params)

        validation_and_test_params = {
            "batch_size": batch_size,
            "shuffle": True,
            "drop_last": False,
        }
        validation_dataset = ClickStreamDataset(validation_dataset.dataset)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, **validation_and_test_params
        )

        test_dataset = ClickStreamDataset(test_dataset.dataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **validation_and_test_params
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

        wiki_dataset.to_csv(
            os.path.join(save_folder, "dataset", "wiki_articles.csv"),
            index=False,
        )
        print("Datasets saved successfully")

    def generate_dataset_sample(
        self,
        sample_size=0.1,
        destination_path="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/click_stream_random_sample.csv",
    ):
        random.seed(123)

        unique_source_articles = (
            self.dataset["source_article"].drop_duplicates().reset_index(drop=True)
        )

        num_articles = len(unique_source_articles)
        sample_dataset_size = int(num_articles * sample_size)

        selected_indices = random.sample(range(num_articles), sample_dataset_size)

        selected_articles = unique_source_articles[
            unique_source_articles.index.isin(selected_indices)
        ]

        dataset_sample = self.dataset[
            self.dataset["source_article"].isin(selected_articles)
        ].reset_index(drop=True)
        dataset_sample.to_csv(destination_path, index=False)

        return dataset_sample


if __name__ == "__main__":
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    creator = DatasetCreation()

    # wiki_documents_dataset = creator.extract_wiki_articles()
    #
    # wiki_documents_dataset.to_csv('../data/complete_wiki_dataset.csv')

    # generate_click_stream_dataset(click_stream_dump_path, wiki_documents_path)

    creator.split_dataset()

    # text_documents = creator.generate_dataset()
    # text_documents.to_csv('../data/wiki_df.csv',
    #                       index=False
    #                       )

    # df = pd.read_csv('../data/wiki_df_complete.csv')
    # get_dataset_sample(df, sample_size=0.05)
