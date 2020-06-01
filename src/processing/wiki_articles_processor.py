import itertools
import json
import pandas as pd
from tqdm import tqdm

from src.utils.utils import clean_title
from ast import literal_eval


class WikiArticlesProcessor:
    def __init__(self, wiki_articles_path):
        self.__wiki_articles_path = wiki_articles_path

        self.__save_path = "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/wiki_articles_english_complete.csv"
        self.__articles = self.extract_wiki_articles()

    def run(self):
        self.__articles.to_csv(self.__save_path, index=False)
        pass

    def get_wiki_articles_data(self):
        return self.__articles

    def get_articles_at_word_level(self):
        return self.__articles["raw_text"].map(self.extract_articles_at_word_level)

    @staticmethod
    def extract_articles_at_word_level(text):
        paragraph_level = list(itertools.chain.from_iterable(literal_eval(text)))
        word_level = list(itertools.chain.from_iterable(paragraph_level))

        return " ".join(word_level)

    def extract_wiki_articles(self):
        text_ids = []
        text_string = []
        links = []
        articles = []

        with open(self.__wiki_articles_path, "r") as json_file:
            json_list = list(json_file)

        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            sections = result["sections"]
            article_text_ids = [section["paragraphs"] for section in sections if section["paragraphs"]]
            article_raw_text = [section["normalized_paragraphs"] for section in sections if section["paragraphs"]]

            # This removes one dimension of the list, keeping the same shape of the paragraphs
            article_text_ids = list(itertools.chain.from_iterable(article_text_ids))
            article_raw_text = list(itertools.chain.from_iterable(article_raw_text))

            text_ids.append(article_text_ids)
            text_string.append(article_raw_text)
            articles.append(clean_title(result["title"]))
            links.append(result["links"])

        wiki_documents_dataset = pd.DataFrame(
            list(zip(articles, text_ids, text_string, links)), columns=["article", "text_ids", "raw_text", "links"]
        )

        return wiki_documents_dataset


if __name__ == "__main__":
    wiki_documents_path = (
        "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/enwiki_entire_article.jsonl"
    )
    wiki = WikiArticlesProcessor(wiki_documents_path)
    wiki.run()
