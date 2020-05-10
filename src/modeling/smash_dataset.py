"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""

import pandas as pd
import numpy as np
import torch
from ast import literal_eval
from torch.utils.data.dataset import Dataset
from typing import List

PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"
TEXT_IDS_COLUMN = "text_ids"
TITLE_COLUMN = "title"


class SMASHDataset(Dataset):
    def __init__(self):
        super(SMASHDataset, self).__init__()

        dataset = pd.read_csv(
            "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/wiki_articles.csv"
        )
        self.text_embeddings = dataset["text_ids"]
        self.articles = dataset["article"]

        max_lengths = self.get_max_lengths()

        self.max_word_length = max_lengths["max_word_length"]
        self.max_sentence_length = max_lengths["max_sentence_length"]
        self.max_paragraph_length = max_lengths["max_paragraph_length"]

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, index):
        text_embedding = self.text_embeddings.iloc[index]

        if isinstance(text_embedding, str):
            text_embedding = literal_eval(self.text_embeddings.iloc[index])

        text_structure = self.get_document_structure(text_embedding)
        words_per_sentence = torch.LongTensor(self.get_padded_words_per_sentence(text_structure["words_per_sentence"]))

        sentences_per_paragraph = torch.LongTensor(
            self.get_padded_sentences_per_paragraph(text_structure["sentences_per_paragraph"])
        )

        paragraphs_per_document = torch.LongTensor([text_structure["paragraphs_per_document"]])

        text_embeddings_padded = torch.LongTensor(self.get_padded_document(text_embedding).astype(np.int64))

        article = {
            "title": self.articles.iloc[index],
            "text_ids": text_embeddings_padded,
            "words_per_sentence": words_per_sentence,
            "sentences_per_paragraph": sentences_per_paragraph,
            "paragraphs_per_document": paragraphs_per_document,
        }

        return article

    def get_articles(self, articles: List[str]):
        """"
        should return a dict of lists following the same structure of article
        article = {
            "title": [self.articles.iloc[index]],
            "text": [text_embeddings_padded],
            "words_per_sentence": [words_per_sentence],
            "sentences_per_paragraph": [sentences_per_paragraph],
            "paragraphs_per_document": [paragraphs_per_document],
        }
        """

        batch_size = 32

        titles_list = []
        text_ids_tensor = torch.LongTensor(
            batch_size, self.max_paragraph_length, self.max_sentence_length, self.max_word_length
        ).zero_()
        words_per_sentence_tensor = torch.LongTensor(
            batch_size, self.max_paragraph_length, self.max_sentence_length
        ).zero_()
        sentences_per_paragraph_tensor = torch.LongTensor(batch_size, self.max_paragraph_length).zero_()
        paragraphs_per_document_tensor = torch.LongTensor(batch_size).zero_()

        for index, article in enumerate(articles):
            article_index = self.articles.index[self.articles == article][0]
            article = self.__getitem__(article_index)
            titles_list.append(article[TITLE_COLUMN])
            text_ids_tensor[index, :] = article[TEXT_IDS_COLUMN]
            words_per_sentence_tensor[index, :] = article[WORDS_PER_SENTENCE_COLUMN]
            sentences_per_paragraph_tensor[index, :] = article[SENTENCES_PER_PARAGRAPH_COLUMN]
            paragraphs_per_document_tensor[index] = article[PARAGRAPHS_PER_DOCUMENT_COLUMN]

        articles_list = {
            TITLE_COLUMN: titles_list,
            TEXT_IDS_COLUMN: text_ids_tensor,
            WORDS_PER_SENTENCE_COLUMN: words_per_sentence_tensor,
            SENTENCES_PER_PARAGRAPH_COLUMN: sentences_per_paragraph_tensor,
            PARAGRAPHS_PER_DOCUMENT_COLUMN: paragraphs_per_document_tensor,
        }

        return articles_list

    @staticmethod
    def get_document_structure(document):
        paragraphs_per_document = len(document)

        sentences_per_paragraph = []
        words_per_sentence = []

        for paragraph in document:
            words_per_sentence_in_paragraph = []

            for sentence in paragraph:
                words_per_sentence_in_paragraph.append(len(sentence))

            words_per_sentence.append(words_per_sentence_in_paragraph)
            sentences_per_paragraph.append(len(paragraph))

        document_structure = {
            "paragraphs_per_document": paragraphs_per_document,
            "sentences_per_paragraph": sentences_per_paragraph,
            "words_per_sentence": words_per_sentence,
        }

        return document_structure

    def get_padded_document(self, document):
        for paragraph in document:
            for sentences in paragraph:
                if len(sentences) < self.max_word_length:
                    extended_words = list([-1] * (self.max_word_length - len(sentences)))
                    sentences.extend(extended_words)

            if len(paragraph) < self.max_sentence_length:
                extended_sentences = list([[-1] * self.max_word_length] * (self.max_sentence_length - len(paragraph)))
                paragraph.extend(extended_sentences)

        if len(document) < self.max_paragraph_length:
            extended_paragraphs = list(
                [[[-1] * self.max_word_length] * self.max_sentence_length] * (self.max_paragraph_length - len(document))
            )
            document.extend(extended_paragraphs)

        document = np.stack(arrays=document, axis=0)
        document += 1

        return document

    def get_padded_words_per_sentence(self, document_structure):
        for paragraph in document_structure:
            if len(paragraph) < self.max_sentence_length:
                extended_sentences = [0 for _ in range(self.max_sentence_length - len(paragraph))]
                paragraph.extend(extended_sentences)

        if len(document_structure) < self.max_paragraph_length:
            extended_paragraphs = [list([0] * self.max_sentence_length)] * (
                self.max_paragraph_length - len(document_structure)
            )

            document_structure.extend(extended_paragraphs)

        document_structure = np.stack(arrays=document_structure, axis=0)

        return document_structure

    def get_padded_sentences_per_paragraph(self, document_structure):
        if len(document_structure) < self.max_paragraph_length:
            extended_paragraphs = [0 for _ in range(self.max_paragraph_length - len(document_structure))]
            document_structure.extend(extended_paragraphs)

        return document_structure

    def get_max_lengths(self):
        word_length_list = []
        sent_length_list = []
        paragraph_length_list = []

        for index, embedding in enumerate(self.text_embeddings):
            for paragraph in literal_eval(embedding):
                for sentences in paragraph:
                    word_length_list.append(len(sentences))

                sent_length_list.append(len(paragraph))

            paragraph_length_list.append(len(literal_eval(embedding)))

        return {
            "max_word_length": max(word_length_list),
            "max_sentence_length": max(sent_length_list),
            "max_paragraph_length": max(paragraph_length_list),
        }


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    test = SMASHDataset()

    print(test.get_articles(["A", "Alan Turing", "Fruit", "Grammar"]))
    # print(test.__getitem__(1))
