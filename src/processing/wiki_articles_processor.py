import sys
import os

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import itertools
import json
import pandas as pd
import torch
from tqdm import tqdm

from utils.utils import clean_title
from ast import literal_eval

from utils.constants import WIKI_ARTICLES_DATASET_PATH, WIKI_ARTICLES_TOKENIZED_PATH


class WikiArticlesProcessor:
    def run(self):
        wiki_articles = self.create_wiki_articles_dataset()
        wiki_articles.to_csv(WIKI_ARTICLES_DATASET_PATH, index=False)

    @staticmethod
    def create_wiki_articles_dataset():
        text_ids = []
        text_string = []
        articles = []

        with open(WIKI_ARTICLES_TOKENIZED_PATH, "r") as json_file:
            json_list = list(json_file)

        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            sections = result["sections"]

            if not sections["paragraphs"]:
                continue

            article_text_ids = [section for section in sections["paragraphs"] if sections["paragraphs"]]
            article_raw_text = [section for section in sections["normalized_paragraphs"] if sections["paragraphs"]]

            # This removes one dimension of the list, keeping the same shape of the paragraphs
            article_text_ids = list(itertools.chain.from_iterable(article_text_ids))
            article_raw_text = list(itertools.chain.from_iterable(article_raw_text))

            if not article_text_ids:
                continue

            text_ids.append(article_text_ids)
            text_string.append(article_raw_text)
            articles.append(clean_title(result["title"]))

        return pd.DataFrame(list(zip(articles, text_ids, text_string)), columns=["article", "text_ids", "raw_text"],)


if __name__ == "__main__":
    os.chdir("/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/")
    wiki = WikiArticlesProcessor()
    wiki.run()
