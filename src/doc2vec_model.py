# Import required libraries
import itertools
import multiprocessing
import os
from ast import literal_eval

import pandas as pd
import torch
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec


class Doc2VecModel:
    def __init__(self,
                 dbow_path='/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/trained_models/doc2vec_dbow_model',
                 dm_path='/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/trained_models/doc2vec_dm_model'):
        self.articles = pd.read_csv(
            '/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/wiki_articles.csv')
        self.articles['raw_text'] = self.articles['raw_text'].map(self.extract_articles_at_word_level)

        self.dbow_path = dbow_path
        self.dm_path = dm_path

        self.model = self.get_model()

    def create_models(self):
        common_model_arguments = dict(
            vector_size=100, epochs=20, min_count=2,
            sample=0, workers=multiprocessing.cpu_count(), negative=5, hs=0,
        )

        dbow_model = Doc2Vec(dm=0, **common_model_arguments)
        dm_model = Doc2Vec(dm=1, dm_concat=1, window=5, **common_model_arguments)

        labeled_questions = self.generate_tagged_documents()

        dbow_model.build_vocab(labeled_questions)
        dm_model.build_vocab(labeled_questions)

        dbow_model.save(self.dbow_path)
        dm_model.save(self.dm_path)

    def get_model(self):
        if not os.path.isfile(self.dbow_path) and not os.path.isfile(self.dm_path):
            self.create_models()

        dbow_model = Doc2Vec.load(self.dbow_path)
        dm_model = Doc2Vec.load(self.dm_path)

        return ConcatenatedDoc2Vec([dbow_model, dm_model])

    def get_entity_vector(self, article: str):
        return torch.Tensor(self.model.docvecs[article])

    def get_hidden_size(self):
        return self.model.models[0].vector_size * 2

    @staticmethod
    def extract_articles_at_word_level(text):
        paragraph_level = list(itertools.chain.from_iterable(literal_eval(text)))
        word_level = list(itertools.chain.from_iterable(paragraph_level))

        return ' '.join(word_level)

    def generate_tagged_documents(self):
        tagged_documents = []

        for i in range(len(self.articles)):
            tagged_documents.append(TaggedDocument(self.articles['raw_text'][i].split(), [self.articles['article'][i]]))

        return tagged_documents


if __name__ == '__main__':
    model = Doc2VecModel()

    print(model.get_entity_vector('Helicopter'))

    # df = df.drop(df[df.isnull().any(axis=1)].index, inplace=True)

    # unique_documents = extract_unique_documents(df)
    # labeled_questions = generate_tagged_documents(unique_documents)
    # model.build_vocab(labeled_questions)
    #
    # current_article_split = generate_split_documents(df[['target_article', 'target_article_raw_text']])
    # previous_article_split = generate_split_documents(df[['source_article', 'source_article_raw_text']])
    #
    # scores = generate_cosine_similarity_scores(current_article_split['raw_text'], previous_article_split['raw_text'])
    #
    # print(scores)

    # # Train the model with 20 epochs
    #
    # for epoch in range(20):
    #     model.train(labeled_questions, epochs=model.iter, total_examples=model.corpus_count)
    #     print("Epoch #{} is complete.".format(epoch + 1))
    #
    # model.most_similar('washington')
    #
    # score = model.n_similarity(questions1_split[i], questions2_split[i])
    #
    # from sklearn.metrics import accuracy_score
    #
    # accuracy = accuracy_score(df.is_duplicate, scores) * 100
