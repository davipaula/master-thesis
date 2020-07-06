"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""

import pandas as pd
import numpy as np
import torch
from ast import literal_eval
from torch.utils.data.dataset import Dataset
from typing import List
import logging
import json

PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"
TEXT_IDS_COLUMN = "text_ids"
TITLE_COLUMN = "title"

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class SMASHDataset(Dataset):
    def __init__(self, dataset_path: str):
        super(SMASHDataset, self).__init__()

        dataset = pd.read_csv(
            dataset_path, usecols=["article", "text_ids"], dtype={"article": "category", "text_ids": "object"}
        )
        self.text_embeddings = dataset["text_ids"].map(json.loads)
        self.articles = dataset["article"]

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, index):
        text_embedding = self.text_embeddings.iloc[index]

        text_structure = self.get_document_structure(text_embedding)

        article = {
            "title": self.articles.iloc[index],
            "text_ids": text_embedding,
            "words_per_sentence": text_structure["words_per_sentence"],
            "sentences_per_paragraph": text_structure["sentences_per_paragraph"],
            "paragraphs_per_document": text_structure["paragraphs_per_document"],
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
        batch_size = len(articles)

        articles_list = []

        for index, article in enumerate(articles):
            article_index = self.articles.index[self.articles == article][0]
            articles_list.append(self.__getitem__(article_index))

        max_lengths = self.get_max_lengths(articles_list)

        max_word_length = max_lengths["max_word_length"]
        max_sentence_length = max_lengths["max_sentence_length"]
        max_paragraph_length = max_lengths["max_paragraph_length"]

        text_ids_tensor = np.zeros([batch_size, max_paragraph_length, max_sentence_length, max_word_length], dtype=int)
        words_per_sentence_tensor = np.zeros([batch_size, max_paragraph_length, max_sentence_length], dtype=int)
        sentences_per_paragraph_tensor = np.zeros([batch_size, max_paragraph_length], dtype=int)
        paragraphs_per_document_tensor = np.zeros(batch_size, dtype=int)

        for index, _ in enumerate(articles):
            article = articles_list[index]
            text_ids_tensor[index, :] = torch.LongTensor(
                self.get_padded_document(
                    article[TEXT_IDS_COLUMN], max_paragraph_length, max_sentence_length, max_word_length
                )
            )
            words_per_sentence_tensor[index, :] = torch.LongTensor(
                self.get_padded_words_per_sentence(
                    article["words_per_sentence"], max_paragraph_length, max_sentence_length
                )
            )
            sentences_per_paragraph_tensor[index, :] = torch.LongTensor(
                self.get_padded_sentences_per_paragraph(article["sentences_per_paragraph"], max_paragraph_length)
            )
            paragraphs_per_document_tensor[index] = torch.LongTensor([article["paragraphs_per_document"]])

        text_ids_tensor = torch.from_numpy(text_ids_tensor)
        words_per_sentence_tensor = torch.from_numpy(words_per_sentence_tensor)
        sentences_per_paragraph_tensor = torch.from_numpy(sentences_per_paragraph_tensor)
        paragraphs_per_document_tensor = torch.from_numpy(paragraphs_per_document_tensor)

        articles_list = {
            TITLE_COLUMN: articles,
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

    @staticmethod
    def get_padded_document(document, max_paragraph_length: int, max_sentence_length: int, max_word_length: int):
        document_placeholder = torch.full(
            (max_paragraph_length, max_sentence_length, max_word_length), -1, dtype=torch.int64
        )

        for paragraph_index, paragraph in enumerate(document):
            for sentence_index, sentence in enumerate(paragraph):
                document_placeholder[paragraph_index, sentence_index, : len(sentence)] = torch.tensor(sentence)

        document_placeholder += 1

        # padded_sentences = np.array(
        #     [
        #         [[row + [-1] * (max_word_length - len(row)) for row in sentence] for sentence in paragraph]
        #         for paragraph in document
        #     ]
        # )
        #
        # padded_document = F.pad(torch.tensor(padded_sentences), (0, 0, 0, 1, 0, 1), "constant", -1)

        return document_placeholder

    @staticmethod
    def get_padded_words_per_sentence(document_structure, max_paragraph_length: int, max_sentence_length: int):
        for paragraph in document_structure:
            if len(paragraph) < max_sentence_length:
                extended_sentences = [0 for _ in range(max_sentence_length - len(paragraph))]
                paragraph.extend(extended_sentences)

        if len(document_structure) < max_paragraph_length:
            extended_paragraphs = [list([0] * max_sentence_length)] * (max_paragraph_length - len(document_structure))

            document_structure.extend(extended_paragraphs)

        document_structure = np.stack(arrays=document_structure, axis=0)

        return document_structure

    def get_padded_sentences_per_paragraph(self, document_structure, max_paragraph_length: int):
        if len(document_structure) < max_paragraph_length:
            extended_paragraphs = [0 for _ in range(max_paragraph_length - len(document_structure))]
            document_structure.extend(extended_paragraphs)

        return document_structure

    def get_max_lengths(self, articles):
        word_length_list = []
        sent_length_list = []
        paragraph_length_list = []

        for index, article in enumerate(articles):
            for paragraph in article["text_ids"]:
                for sentences in paragraph:
                    word_length_list.append(len(sentences))

                sent_length_list.append(len(paragraph))

            paragraph_length_list.append(len(article["text_ids"]))

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

    test = SMASHDataset(
        "/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/dataset/wiki_articles_english.csv"
    )

    articles_to_get = [
        "Anarchism",
        "Autism",
        "A",
        "Achilles",
        "Abraham Lincoln",
        "Aristotle",
        "An American in Paris",
        "Academy Awards",
        "Ayn Rand",
        "Algeria",
        "Anthropology",
        "Alchemy",
        "ASCII",
        "Apollo",
        "Andre Agassi",
        "Andorra",
        "Amphibian",
        "Alaska",
        "Agriculture",
        "Algae",
        "Analysis of variance",
        "Apollo 11",
        "Apollo 8",
        "Astronaut",
        "Alphabet",
        "Atomic number",
        "Asia",
        "Articles of Confederation",
        "Atlantic Ocean",
        "Angola",
        "Alberta",
        "Albert Einstein",
    ]

    logger.info("Started getting articles")

    for n in range(10):
        (test.get_articles(articles_to_get))

    logger.info("Finished getting articles")
    # print(test.__getitem__(1))
