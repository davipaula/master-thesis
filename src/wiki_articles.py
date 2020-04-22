import itertools
from ast import literal_eval

import pandas as pd


class WikiArticles:
    def __init__(self, limit_rows: int):
        self.articles = pd.read_csv(
            '/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/wiki_articles.csv', nrows=50)

    def get_articles_at_word_level(self):
        return self.articles['raw_text'].map(self.extract_articles_at_word_level)

    @staticmethod
    def extract_articles_at_word_level(text):
        paragraph_level = list(itertools.chain.from_iterable(literal_eval(text)))
        word_level = list(itertools.chain.from_iterable(paragraph_level))

        return ' '.join(word_level)
