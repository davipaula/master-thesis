"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import torch
from typing import List
from wikipedia2vec import Wikipedia2Vec
from src.utils.utils import cosine_similarity

MODEL_FILE = "./trained_models/wikipedia2vec_model"


class Wikipedia2VecModel:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        self.model = Wikipedia2Vec.load(MODEL_FILE)

    def get_entity_vector(self, articles: List[str]):
        entity_vectors = torch.zeros((len(articles), 100), dtype=torch.float, device=self.device)

        for article_index, article in enumerate(articles):
            try:
                entity_vectors[article_index, :] = torch.from_numpy(self.model.get_entity_vector(article))
            except:
                entity_vectors[article_index, :] = torch.zeros(100)

        return entity_vectors

    def get_hidden_size(self):
        return 100

    def calculate_cosine_pair_documents(self, current_article_title, previous_article_title):
        current_article_vector = self.model.get_entity_vector(current_article_title)
        previous_article_vector = self.model.get_entity_vector(previous_article_title)

        return cosine_similarity(current_article_vector, previous_article_vector)


if __name__ == "__main__":
    model = Wikipedia2VecModel()
