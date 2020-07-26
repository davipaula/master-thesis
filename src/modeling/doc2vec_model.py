import itertools
import logging
import multiprocessing
import os
import torch
from ast import literal_eval
from src.data_structure.wiki_articles_dataset import WikiArticlesDataset
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from typing import List

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

TRAIN_DATASET_PATH = "./data/dataset/click_stream_train.pth"

ARTICLE_COLUMN = "article"
RAW_TEXT_COLUMN = "raw_text"
TRAIN_FOLD = "train"
FOLD_COLUMN = "fold"


class Doc2VecModel:
    def __init__(
        self, dbow_path="./trained_models/doc2vec_dbow_model", dm_path="./trained_models/doc2vec_dm_model",
    ):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        self.__articles = WikiArticlesDataset().get_articles(level="word")

        self.dbow_path = dbow_path
        self.dm_path = dm_path

        self.model = self.get_model()

    def create_models(self):
        common_model_arguments = dict(
            vector_size=100, epochs=20, min_count=2, sample=0, workers=multiprocessing.cpu_count(), negative=5, hs=0,
        )

        dbow_model = Doc2Vec(dm=0, **common_model_arguments)
        dm_model = Doc2Vec(dm=1, dm_concat=1, window=5, **common_model_arguments)

        labeled_questions = self.generate_tagged_documents()

        logger.info("Building vocab DBOW")
        dbow_model.build_vocab(labeled_questions)

        logger.info("Building vocab DM")
        dm_model.build_vocab(labeled_questions)

        dbow_model.save(self.dbow_path)
        dm_model.save(self.dm_path)

    def get_model(self):
        if not os.path.isfile(self.dbow_path) and not os.path.isfile(self.dm_path):
            self.create_models()

        dbow_model = Doc2Vec.load(self.dbow_path)
        dm_model = Doc2Vec.load(self.dm_path)

        return ConcatenatedDoc2Vec([dbow_model, dm_model])

    def get_entity_vector(self, articles: List[str]):
        entity_vectors = torch.zeros((len(articles), self.get_hidden_size()), dtype=torch.float, device=self.device)

        for article_index, article in enumerate(articles):
            entity_vectors[article_index] = torch.from_numpy(self.model.docvecs[article])

        return entity_vectors

    def get_inferred_vector(self, articles: List[str]):
        inferred_vectors = torch.zeros((len(articles), self.get_hidden_size()), dtype=torch.float, device=self.device)

        for article_index, article in enumerate(articles):
            article_raw_text = self.__articles[RAW_TEXT_COLUMN][self.__articles[ARTICLE_COLUMN] == article]

            inferred_vectors[article_index] = torch.from_numpy(self.model.infer_vector(article_raw_text))

        return inferred_vectors

    def get_hidden_size(self):
        return self.model.models[0].vector_size * 2

    @staticmethod
    def extract_articles_at_word_level(text):
        paragraph_level = list(itertools.chain.from_iterable(literal_eval(text)))
        word_level = list(itertools.chain.from_iterable(paragraph_level))

        return " ".join(word_level)

    def generate_tagged_documents(self):
        tagged_documents = []
        train_documents = []

        click_stream_train_dataset = torch.load(TRAIN_DATASET_PATH)
        train_articles_titles = list(
            set(
                (
                    click_stream_train_dataset.dataset["source_article"].tolist()
                    + click_stream_train_dataset.dataset["target_article"].tolist()
                )
            )
        )

        train_articles = self.__articles[self.__articles[ARTICLE_COLUMN].isin(train_articles_titles)]

        for i in range(len(train_articles)):
            tagged_documents.append(
                TaggedDocument(
                    train_articles[RAW_TEXT_COLUMN].iloc[i].split(), [train_articles[ARTICLE_COLUMN].iloc[i]],
                )
            )

        return tagged_documents


if __name__ == "__main__":
    model = Doc2VecModel()

    model.create_models()
