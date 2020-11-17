import pandas as pd
import random
import torch
from ast import literal_eval


class NegativeSamplerProcessor:
    def __init__(self, click_stream_dataset, wiki_articles_dataset):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        random.seed(123)

        self.click_stream_dataset = click_stream_dataset
        self.wiki_articles_dataset = wiki_articles_dataset

    def add_negative_sampling(self):
        source_articles = (
            pd.merge(
                self.click_stream_dataset[["source_article"]],
                self.wiki_articles_dataset[["article", "links"]],
                left_on=["source_article"],
                right_on=["article"],
            )
            .drop(["article"], axis=1)
            .drop_duplicates(subset="source_article")
        )

        source_articles["links"] = source_articles["links"].map(literal_eval)
        source_articles = source_articles.explode("links")
        visited_articles = self.click_stream_dataset[
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

        self.click_stream_dataset.append(negative_sampling, ignore_index=True)

        return self.click_stream_dataset
