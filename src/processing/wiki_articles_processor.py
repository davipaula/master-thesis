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

SAVE_PATH = "./data/processed/wiki_articles_english_complete.csv"
WIKI_DOCUMENTS_PATH = "./data/processed/enwiki_tokenized_selected_articles.jsonl"


class WikiArticlesProcessor:
    def __init__(self, wiki_articles_path):
        self.__wiki_articles_path = wiki_articles_path

        self.__save_path = SAVE_PATH
        self.__articles = self.load_wiki_articles()

    def run(self):
        self.__articles.to_csv(self.__save_path, index=False)

    def get_wiki_articles_data(self):
        return self.__articles

    def get_articles_at_word_level(self):
        return self.__articles["raw_text"].map(self.extract_articles_at_word_level)

    @staticmethod
    def extract_articles_at_word_level(text):
        paragraph_level = list(itertools.chain.from_iterable(literal_eval(text)))
        word_level = list(itertools.chain.from_iterable(paragraph_level))

        return " ".join(word_level)

    def load_wiki_articles(self):
        text_ids = []
        text_string = []
        articles = []
        articles_to_remove = []

        with open(self.__wiki_articles_path, "r") as json_file:
            json_list = list(json_file)

        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            sections = result["sections"]

            if result["title"] == "Lage Raho Munna Bhai":
                print("Check here")

            if not sections["paragraphs"]:
                articles_to_remove.append(result["title"])

                continue

            article_text_ids = [section for section in sections["paragraphs"] if sections["paragraphs"]]
            article_raw_text = [section for section in sections["normalized_paragraphs"] if sections["paragraphs"]]

            # This removes one dimension of the list, keeping the same shape of the paragraphs
            article_text_ids = list(itertools.chain.from_iterable(article_text_ids))
            article_raw_text = list(itertools.chain.from_iterable(article_raw_text))

            text_ids.append(article_text_ids)
            text_string.append(article_raw_text)
            articles.append(clean_title(result["title"]))

            pd.Series(articles_to_remove).to_csv("./data/processed/articles_to_remove.txt", header=False, index=False)

        return pd.DataFrame(list(zip(articles, text_ids, text_string)), columns=["article", "text_ids", "raw_text"],)


if __name__ == "__main__":
    wiki = WikiArticlesProcessor(WIKI_DOCUMENTS_PATH)
    wiki.run()
