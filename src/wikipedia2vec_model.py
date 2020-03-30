from wikipedia2vec import Wikipedia2Vec
from utils import cosine_similarity


class Wikipedia2VecModel:
    def __init__(self):
        model_file = '../data/enwiki_20180420_lg1_300d.pkl'

        self.model = Wikipedia2Vec.load(model_file)

    def calculate_cosine_pair_documents(self, current_article_title, previous_article_title):
        current_article_vector = self.model.get_entity_vector(current_article_title)
        previous_article_vector = self.model.get_entity_vector(previous_article_title)

        return cosine_similarity(current_article_vector, previous_article_vector)
