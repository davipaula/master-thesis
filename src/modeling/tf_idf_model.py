"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import itertools
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfIdfModel:
    def __init__(self):
        self.articles = pd.read_csv('../data/wiki_articles.csv', nrows=500)
        self.articles['raw_text'] = self.articles['raw_text'].map(self.extract_articles_at_word_level)

        self.tf_idf_matrix = self.build_tf_idf_matrix(self.articles['raw_text'])

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
    def extract_articles_at_word_level(text):
        paragraph_level = list(itertools.chain.from_iterable(literal_eval(text)))
        word_level = list(itertools.chain.from_iterable(paragraph_level))

        return ' '.join(word_level)

    @staticmethod
    def build_tf_idf_matrix(documents):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)

        return tf.fit_transform(documents)

    def find_similar(self, index, top_n=5):
        cosine_similarities = linear_kernel(self.tf_idf_matrix[index:index + 1], self.tf_idf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]

        return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    def calculate_cosine_pair_documents(self, target_article_title: str, source_article_title: str):
        target_article_index = self.articles.loc[self.articles['article'] == target_article_title].index[0]
        source_article_index = self.articles.loc[self.articles['article'] == source_article_title].index[0]

        target_article_cosine_similarities = linear_kernel(
            self.tf_idf_matrix[target_article_index:target_article_index + 1], self.tf_idf_matrix).flatten()

        return target_article_cosine_similarities[source_article_index]


if __name__ == '__main__':
    model = TfIdfModel()
    similarity = model.calculate_cosine_pair_documents('Farm', 'Food')
    print(similarity)
