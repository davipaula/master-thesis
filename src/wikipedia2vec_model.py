import torch
from wikipedia2vec import Wikipedia2Vec

from utils import cosine_similarity


class Wikipedia2VecModel:
    def __init__(self):
        model_file = './trained_models/wikipedia2vec_model'

        self.model = Wikipedia2Vec.load(model_file)

    def get_entity_vector(self, article: str):
        try:
            return torch.Tensor(self.model.get_entity_vector(article))
        except:
            return torch.zeros(100)

    def get_hidden_size(self):
        return 100

    def calculate_cosine_pair_documents(self, current_article_title, previous_article_title):
        current_article_vector = self.model.get_entity_vector(current_article_title)
        previous_article_vector = self.model.get_entity_vector(previous_article_title)

        return cosine_similarity(current_article_vector, previous_article_vector)


if __name__ == '__main__':
    model = Wikipedia2VecModel()
