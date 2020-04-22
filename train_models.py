import pandas as pd
import torch
from torch import nn

from doc2vec_model import Doc2VecModel
from wikipedia2vec_model import Wikipedia2VecModel


class TrainModels:
    def __init__(self):
        self.articles = pd.read_csv('./data/wiki_articles.csv')
        self.click_stream = pd.read_csv('./data/click_stream.csv', nrows=1000)
        self.doc2vec = Doc2VecModel()
        self.wikipedia2vec = Wikipedia2VecModel()

        self.models = {}
        self.models = {'doc2vec': self.doc2vec,
                       'wikipedia2vec': self.wikipedia2vec}

        self.num_epochs = 5

        self.criterion = nn.SmoothL1Loss()

    def get_optimizer(self, regression_model):
        learning_rate = 10e-5
        return torch.optim.Adam(regression_model.parameters(),
                                lr=learning_rate)

    def get_regression_model(self, hidden_size):
        input_dim = hidden_size * 3  # 3 = number of concatenations
        # Not mentioned in the paper.
        mlp_dim = int(input_dim / 2)
        output_dim = 1

        return nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim),
            nn.Sigmoid()
        )

    def train(self):
        for model_name, model in self.models.items():
            regression_model = self.get_regression_model(model.get_hidden_size())
            optimizer = self.get_optimizer(regression_model)

            for epoch in range(self.num_epochs):
                regression_model.train()

                for click_stream_index in range(len(self.click_stream)):
                    source_article = self.click_stream['source_article'].iloc[click_stream_index]
                    target_article = self.click_stream['target_article'].iloc[click_stream_index]
                    click_rate = torch.Tensor([self.click_stream['click_rate'].iloc[click_stream_index]])

                    source_article_vector = model.get_entity_vector(source_article)
                    target_article_vector = model.get_entity_vector(target_article)

                    optimizer.zero_grad()

                    siamese_representation = self.get_siamese_representation(source_article_vector,
                                                                             target_article_vector)

                    prediction = regression_model(siamese_representation)

                    loss = self.criterion(prediction, click_rate)
                    loss.backward()
                    optimizer.step()

                    if click_stream_index == 10:
                        print(loss)
                        break

            regression_model.save('./trained_models/{}_regression_model'.format(str(model_name)))

    @staticmethod
    def get_siamese_representation(source_document, target_document):
        return torch.cat((source_document,
                          target_document,
                          torch.abs(source_document - target_document),
                          ), 0)


if __name__ == '__main__':
    train = TrainModels()
    train.train()
