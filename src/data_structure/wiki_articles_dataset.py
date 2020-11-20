import itertools
from ast import literal_eval

import pandas as pd

from ..utils.constants import ARTICLE_COLUMN, RAW_TEXT_COLUMN


class WikiArticlesDataset:
    def __init__(self):
        __wiki_articles_path = "./data/dataset/wiki_articles_english_complete.csv"

        __wiki_articles = pd.read_csv(__wiki_articles_path)
        self.dataset = __wiki_articles

    def __len__(self):
        return len(self.dataset)

    def get_available_titles(self):
        return self.dataset[ARTICLE_COLUMN].unique()

    def get_articles(self, level="paragraph"):
        train_articles = self.dataset.copy()

        if level == "word":
            train_articles[RAW_TEXT_COLUMN] = train_articles[RAW_TEXT_COLUMN].map(
                self.extract_articles_at_word_level
            )
        elif level == "sentence":
            train_articles[RAW_TEXT_COLUMN] = train_articles[RAW_TEXT_COLUMN].map(
                self.extract_articles_at_sentence_level
            )

        return train_articles

    @staticmethod
    def extract_articles_at_word_level(text):
        sentence_level = list(itertools.chain.from_iterable(literal_eval(text)))
        word_level = list(itertools.chain.from_iterable(sentence_level))

        return " ".join(word_level)

    @staticmethod
    def extract_articles_at_sentence_level(text):
        sentence_level = list(itertools.chain.from_iterable(literal_eval(text)))

        return " ".join(sentence_level)
