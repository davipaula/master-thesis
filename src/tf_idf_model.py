from ast import literal_eval
from random import randint
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from utils import cosine_similarity


class TfIdfModel:
    def __init__(self):
        df = pd.read_csv('../data/wiki_df_text.csv', nrows=500)
        self.documents = self.extract_unique_documents(df)

        self.tf_idf_matrix = self.build_tf_idf_matrix(self.documents['text'])

    @staticmethod
    def extract_unique_documents(dataset):
        current_texts = dataset[['current_article', 'current_article_text']]
        previous_texts = dataset[['previous_article', 'previous_article_text']]

        current_texts.columns = ['title', 'text']
        previous_texts.columns = ['title', 'text']

        combined_texts = current_texts.append(previous_texts, sort=False, ignore_index=True)

        unique_documents = combined_texts.drop_duplicates().reset_index(drop=True)

        return unique_documents

    @staticmethod
    def build_tf_idf_matrix(documents):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)

        return tf.fit_transform(documents)

    def find_similar(self, index, top_n=5):
        cosine_similarities = linear_kernel(self.tf_idf_matrix[index:index + 1], self.tf_idf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]

        return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    def calculate_cosine_pair_documents(self, current_article_title: str, previous_article_title: str):
        current_article_index = self.documents.loc[self.documents['title'] == current_article_title].index[0]
        previous_article_index = self.documents.loc[self.documents['title'] == previous_article_title].index[0]

        current_article_cosine_similarities = linear_kernel(
            self.tf_idf_matrix[current_article_index:current_article_index + 1], self.tf_idf_matrix).flatten()

        return current_article_cosine_similarities[previous_article_index]