import pandas as pd
import torch
from torch import nn

from doc2vec_model import Doc2VecModel
from wikipedia2vec_model import Wikipedia2VecModel


class TrainModels:
    def __init__(self):
        self.articles = torch.load(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/wiki_articles_train.pth"
        )
        self.click_stream = torch.load(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/click_stream_train.pth"
        )
        self.doc2vec = Doc2VecModel()
        self.wikipedia2vec = Wikipedia2VecModel()

        self.models = {}
        self.models = {"doc2vec": self.doc2vec, "wikipedia2vec": self.wikipedia2vec}

        self.num_epochs = 5

        self.criterion = nn.SmoothL1Loss()

    def get_optimizer(self, regression_model):
        learning_rate = 10e-5
        return torch.optim.Adam(regression_model.parameters(), lr=learning_rate)

    def get_regression_model(self, hidden_size):
        input_dim = hidden_size * 3  # 3 = number of concatenations
        # Not mentioned in the paper.
        mlp_dim = int(input_dim / 2)
        output_dim = 1

        return nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim),
            nn.Sigmoid(),
        )

    def train(self):
        for model_name, model in self.models.items():
            regression_model = self.get_regression_model(model.get_hidden_size())
            optimizer = self.get_optimizer(regression_model)

            for epoch in range(self.num_epochs):
                regression_model.train()

                i = 0

                for row in self.click_stream:
                    source_article_vector = model.get_entity_vector(
                        row["source_article"]
                    )
                    target_article_vector = model.get_entity_vector(
                        row["target_article"]
                    )

                    optimizer.zero_grad()

                    siamese_representation = self.get_siamese_representation(
                        source_article_vector, target_article_vector
                    )

                    prediction = regression_model(siamese_representation)

                    loss = self.criterion(prediction, row["click_rate"])
                    loss.backward()
                    optimizer.step()

                    if i == 2:
                        print(loss)
                        break

                    i += 1

            torch.save(
                regression_model,
                "./trained_models/{}_regression_model".format(str(model_name)),
            )

    @staticmethod
    def get_siamese_representation(source_document, target_document):
        return torch.cat(
            (
                source_document,
                target_document,
                torch.abs(source_document - target_document),
            ),
            1,
        )


if __name__ == "__main__":
    train = TrainModels()
    train.train()
