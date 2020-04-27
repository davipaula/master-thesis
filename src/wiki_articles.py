import itertools
import json
import os
from ast import literal_eval

import pandas as pd

from utils import clean_title


class WikiArticlesProcessor:
    def __init__(self, wiki_articles_path):
        self.__wiki_articles_path = wiki_articles_path

        save_path = '/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/wiki_articles.csv'
        if os.path.exists(save_path):
            self.__articles = pd.read_csv(save_path)
        else:
            self.__articles = self.extract_wiki_articles()
            self.__articles.to_csv(save_path, index=False)

        print('Saved wiki articles')

    def get_wiki_articles_data(self):
        return self.__articles

    def get_articles_at_word_level(self):
        return self.__articles['raw_text'].map(self.extract_articles_at_word_level)

    @staticmethod
    def extract_articles_at_word_level(text):
        paragraph_level = list(itertools.chain.from_iterable(literal_eval(text)))
        word_level = list(itertools.chain.from_iterable(paragraph_level))

        return ' '.join(word_level)

    def extract_wiki_articles(self):
        text_ids = []
        text_string = []
        links = []
        articles = []

        with open(self.__wiki_articles_path, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            introduction_ids = result['sections'][0]['paragraphs']
            # Cleaning empty paragraphs and sentences
            sentences_to_append = [filtered_sentence for filtered_sentence in
                                   [[sentence for sentence in paragraph if sentence]
                                    for paragraph in introduction_ids if paragraph] if filtered_sentence]

            introduction_string = result['sections'][0]['normalized_paragraphs']

            if sentences_to_append:
                text_ids.append(sentences_to_append)
                text_string.append(introduction_string)
                articles.append(clean_title(result['title']))
                links.append(result['links'])

        wiki_documents_dataset = pd.DataFrame(list(zip(articles, text_ids, text_string, links)),
                                              columns=['article', 'text_ids', 'raw_text', 'links'])

        return wiki_documents_dataset

    def get_available_titles(self):
        return self.__articles['article'].unique()


if __name__ == '__main__':
    wiki_documents_path = '/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/simplewiki3.jsonl'
    wiki = WikiArticlesProcessor(wiki_documents_path)
