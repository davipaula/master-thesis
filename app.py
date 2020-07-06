import sys
import os

src_path = os.path.join(os.getcwd(), "src")
sys.path.extend([os.getcwd(), src_path])

import torch
from fastapi import FastAPI
from torch import nn

from modeling.doc2vec_model import Doc2VecModel

app = FastAPI()

model = Doc2VecModel()
regression_model = torch.load("./trained_models/doc2vec_regression_model")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/newEndpoint2")
def new_endpoint():
    return {"This is a new": "Endpoint2"}


@app.get("/predict/")
def predict(source_article: str, target_article: str):
    source_article_vector = model.get_inferred_vector([source_article])
    target_article_vector = model.get_inferred_vector([target_article])

    siamese_representation = get_siamese_representation(source_article_vector, target_article_vector)

    prediction = regression_model(siamese_representation)

    return f"Predicted click rate: {prediction.data}"


def get_siamese_representation(source_article_vector, target_article_vector):
    return torch.cat(
        (source_article_vector, target_article_vector, torch.abs(source_article_vector - target_article_vector),), 1,
    )
