"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing import List, Dict
import logging
import json

PARAGRAPHS_PER_DOCUMENT_COLUMN = "paragraphs_per_document"
SENTENCES_PER_PARAGRAPH_COLUMN = "sentences_per_paragraph"
WORDS_PER_SENTENCE_COLUMN = "words_per_sentence"
TEXT_IDS_COLUMN = "text_ids"
TITLE_COLUMN = "title"

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class SMASHDataset(Dataset):
    def __init__(self, dataset_path: str, introduction_only: bool = False):
        """

        Parameters
        ----------
        dataset_path : str
            Path where Wikipedia articles dataset is stored
        introduction_only : bool
            Flag to use only introduction section of the articles
        """
        super(SMASHDataset, self).__init__()

        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        dataset = pd.read_csv(
            dataset_path,
            usecols=["article", "text_ids", "text_ids_intro"],
            dtype={
                "article": "category",
                "text_ids": "object",
                "text_ids_intro": "object",
            },
        )

        if introduction_only:
            self.text_embeddings = dataset["text_ids_intro"].map(json.loads)
        else:
            self.text_embeddings = dataset["text_ids"].map(json.loads)

        self.articles = dataset["article"]

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, index: np.int64) -> Dict[str, str]:
        """Returns the data for an article

        Parameters
        ----------
        index : np.int64
            Index that represents the article

        Returns
        -------
        Dict[str, str]

        """
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

    def get_articles(self, articles: List[str]) -> Dict[str, List[str]]:
        """ "
        Returns a dictionary with data for the article, according to the structure below:

        article = {
            "title": [self.articles.iloc[index]],
            "text": [text_embeddings_padded],
            "words_per_sentence": [words_per_sentence],
            "sentences_per_paragraph": [sentences_per_paragraph],
            "paragraphs_per_document": [paragraphs_per_document],
        }

        Parameters
        ----------
        articles : List[str]
            Title of the articles to be retrieved

        Returns
        -------
        Dict[str, List[str]]
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

        text_ids_tensor = torch.zeros(
            [batch_size, max_paragraph_length, max_sentence_length, max_word_length],
            dtype=int,
            device=self.device,
        )
        words_per_sentence_tensor = torch.zeros(
            [batch_size, max_paragraph_length, max_sentence_length],
            dtype=int,
            device=self.device,
        )
        sentences_per_paragraph_tensor = torch.zeros(
            [batch_size, max_paragraph_length], dtype=int, device=self.device
        )
        paragraphs_per_document_tensor = torch.zeros(
            batch_size, dtype=int, device=self.device
        )

        for index, _ in enumerate(articles):
            article = articles_list[index]
            text_ids_tensor[index, :] = (
                self.get_padded_article(
                    article[TEXT_IDS_COLUMN],
                    max_paragraph_length,
                    max_sentence_length,
                    max_word_length,
                )
                .detach()
                .clone()
            )

            words_per_sentence_tensor[index, :] = torch.tensor(
                self.get_padded_words_per_sentence(
                    article["words_per_sentence"],
                    max_paragraph_length,
                    max_sentence_length,
                ),
                device=self.device,
                dtype=torch.int,
            )
            sentences_per_paragraph_tensor[index, :] = torch.tensor(
                self.get_padded_sentences_per_paragraph(
                    article["sentences_per_paragraph"], max_paragraph_length
                ),
                device=self.device,
                dtype=torch.int,
            )
            paragraphs_per_document_tensor[index] = torch.tensor(
                [article["paragraphs_per_document"]],
                device=self.device,
                dtype=torch.int,
            )

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

    def get_padded_article(
        self,
        article: List[List[List[int]]],
        max_paragraph_length: int,
        max_sentence_length: int,
        max_word_length: int,
    ) -> torch.Tensor:
        """Returns a zero padded article

        Parameters
        ----------
        article : List[List[List[int]]]
            Batch of articles
        max_paragraph_length : int
            Length of document with the largest number of paragraphs in batch
        max_sentence_length : int
            Length of the paragraph with the largest number of sentences in batch
        max_word_length : int
            Length of the sentence with the largest number of words in batch

        Returns
        -------
        torch.Tensor

        """
        document_placeholder = torch.full(
            (max_paragraph_length, max_sentence_length, max_word_length),
            -1,
            dtype=torch.int,
            device=self.device,
        )

        for paragraph_index, paragraph in enumerate(article):
            for sentence_index, sentence in enumerate(paragraph):
                document_placeholder[
                    paragraph_index, sentence_index, : len(sentence)
                ] = (
                    torch.tensor(sentence, dtype=torch.int, device=self.device)
                    .clone()
                    .detach()
                )

        document_placeholder += 1

        return document_placeholder

    @staticmethod
    def get_padded_words_per_sentence(
        document_structure, max_paragraph_length: int, max_sentence_length: int
    ):
        for paragraph in document_structure:
            if len(paragraph) < max_sentence_length:
                extended_sentences = [
                    0 for _ in range(max_sentence_length - len(paragraph))
                ]
                paragraph.extend(extended_sentences)

        if len(document_structure) < max_paragraph_length:
            extended_paragraphs = [list([0] * max_sentence_length)] * (
                max_paragraph_length - len(document_structure)
            )

            document_structure.extend(extended_paragraphs)

        document_structure = np.stack(arrays=document_structure, axis=0)

        return document_structure

    @staticmethod
    def get_padded_sentences_per_paragraph(
        document_structure: List[int], max_paragraph_length: int
    ) -> List[int]:
        """Returns a padded sentences per paragraph

        Parameters
        ----------
        document_structure : List[int]
            Document structure
        max_paragraph_length : int
            Length of the largest paragraph

        Returns
        -------
        List[int]

        """
        if len(document_structure) < max_paragraph_length:
            extended_paragraphs = [
                0 for _ in range(max_paragraph_length - len(document_structure))
            ]
            document_structure.extend(extended_paragraphs)

        return document_structure

    @staticmethod
    def get_max_lengths(articles: List[Dict[str, str]]) -> Dict[str, int]:
        """Get the max length in a batch of articles

        Parameters
        ----------
        articles : List[Dict[str, str]]
            Batch of articles

        Returns
        -------
        Dict[str, int]

        """
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

    def get_n_percentile_paragraph_length(self, n: int = 90) -> int:
        """Gets the length of the paragraph at the nth percentile

        Parameters
        ----------
        n : int
            Percentile to get the paragraph length

        Returns
        -------
        int

        """
        articles_text = self.text_embeddings
        paragraphs_per_article = [len(article_text) for article_text in articles_text]

        return int(np.percentile(paragraphs_per_article, n))
