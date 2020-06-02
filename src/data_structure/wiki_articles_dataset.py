import itertools
from ast import literal_eval

import pandas as pd
from torch.utils.data.dataset import Dataset

FOLD_COLUMN = "fold"
ARTICLE_COLUMN = "article"
RAW_TEXT_COLUMN = "raw_text"


class WikiArticlesDataset:
    def __init__(self):
        __selected_articles_path = "./data/processed/selected_articles.csv"
        __wiki_articles_path = "./data/dataset/wiki_articles_english.csv"

        __selected_articles = pd.read_csv(__selected_articles_path)
        __wiki_articles = pd.read_csv(__wiki_articles_path)
        self.dataset = pd.merge(__selected_articles, __wiki_articles, on=["article"])

    def __len__(self):
        return len(self.dataset)

    def get_available_titles(self):
        return self.dataset[ARTICLE_COLUMN].unique()

    def get_articles(self, level="paragraph"):
        train_articles = self.dataset.copy()

        if level == "word":
            train_articles[RAW_TEXT_COLUMN] = train_articles[RAW_TEXT_COLUMN].map(self.extract_articles_at_word_level)
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


if __name__ == "__main__":
    data = WikiArticlesDataset()
    print(data.get_articles())
