import itertools
from ast import literal_eval

import pandas as pd
from torch.utils.data.dataset import Dataset


class WikiArticlesDataset:
    def __init__(self):
        super(WikiArticlesDataset, self).__init__()

        __save_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/wiki_articles.csv"
        self.dataset = pd.read_csv(__save_path)

    def __len__(self):
        return len(self.dataset)

    def get_available_titles(self):
        return self.dataset["article"].unique()

    def get_train_articles(self, level="paragraph"):
        train_articles = self.dataset[self.dataset["fold"] == "train"].copy()

        if level == "word":
            train_articles["raw_text"] = train_articles["raw_text"].map(self.extract_articles_at_word_level)
        elif level == "sentence":
            train_articles["raw_text"] = train_articles["raw_text"].map(self.extract_articles_at_sentence_level)

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
    print(data.get_train_articles())
