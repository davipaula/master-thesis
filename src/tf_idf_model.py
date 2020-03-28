from ast import literal_eval
from random import randint

from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

"""
- Opens files
- Build TF-IDF matrix for each file
"""


def extract_unique_documents(dataset):
    current_texts = dataset[['current_article', 'current_article_text']]
    previous_texts = dataset[['previous_article', 'previous_article_text']]

    current_texts.columns = ['title', 'text']
    previous_texts.columns = ['title', 'text']

    combined_texts = current_texts.append(previous_texts, sort=False, ignore_index=True)

    unique_documents = combined_texts.drop_duplicates().reset_index(drop=True)

    return unique_documents


def build_tf_idf_matrix(document):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)

    return tf.fit_transform(document)


def find_similar(document, index, top_n=5):
    tf_idf_matrix = build_tf_idf_matrix(document['text'])

    cosine_similarities = linear_kernel(tf_idf_matrix[index:index + 1], tf_idf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]

    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]


def execute_tf_idf():
    df = pd.read_csv('../data/wiki_df_text.csv', nrows=500)
    documents = extract_unique_documents(df)

    selected_text_index = randint(0, len(documents))

    print(documents['title'].iloc[selected_text_index])

    for index, score in find_similar(documents, selected_text_index):
        print(score, documents['title'].iloc[index])


if __name__ == '__main__':
    execute_tf_idf()
