"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import gc
import logging
import os
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from torch.nn import Softmax

from utils.constants import (
    TEXT_IDS_COLUMN,
    WORDS_PER_SENTENCE_COLUMN,
    SENTENCES_PER_PARAGRAPH_COLUMN,
    PARAGRAPHS_PER_DOCUMENT_COLUMN,
)
from utils.utils import (
    remove_zeros,
    get_words_per_document_at_word_level,
    get_document_at_word_level,
    get_document_at_sentence_level,
    get_words_per_sentence_at_sentence_level,
    get_sentences_per_paragraph_at_sentence_level,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

GRU_HIDDEN_SIZE = 128
HIDDEN_LAYER_SIZE = 128

LEVELS = {
    "paragraph": ["paragraph"],
    "sentence": ["paragraph", "sentence"],
    "word": ["paragraph", "sentence", "word"],
}


class SmashRNNModel(nn.Module):
    def __init__(self, dict, dict_len, embedding_size, levels):
        super(SmashRNNModel, self).__init__()

        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(123)
            self.device = torch.device("cpu")

        self.dict = dict

        # Init embedding layer
        self.embedding = (
            nn.Embedding(num_embeddings=dict_len, embedding_dim=embedding_size)
            .from_pretrained(dict)
            .to(self.device)
        )

        # RNN + attention layers
        word_gru_hidden_size = GRU_HIDDEN_SIZE
        self.word_gru_out_size = word_gru_hidden_size * 2
        self.word_gru = nn.GRU(
            embedding_size, word_gru_hidden_size, bidirectional=True, batch_first=True
        ).to(self.device)
        self.word_attention = nn.Linear(
            self.word_gru_out_size, self.word_gru_out_size
        ).to(self.device)
        self.word_context_layer = nn.Linear(self.word_gru_out_size, 1, bias=False).to(
            self.device
        )  # Word context vector to take dot-product with

        sentence_gru_hidden_size = GRU_HIDDEN_SIZE
        self.sentence_gru_out_size = sentence_gru_hidden_size * 2
        self.sentence_gru = nn.GRU(
            self.word_gru_out_size,
            sentence_gru_hidden_size,
            bidirectional=True,
            batch_first=True,
        ).to(self.device)
        self.sentence_attention = nn.Linear(
            self.sentence_gru_out_size, self.sentence_gru_out_size
        ).to(self.device)
        self.sentence_context_layer = nn.Linear(
            self.sentence_gru_out_size, 1, bias=False
        ).to(self.device)

        paragraph_gru_hidden_size = GRU_HIDDEN_SIZE
        self.paragraph_gru_out_size = paragraph_gru_hidden_size * 2
        self.paragraph_gru = nn.GRU(
            self.sentence_gru_out_size,
            paragraph_gru_hidden_size,
            bidirectional=True,
            batch_first=True,
        ).to(self.device)
        self.paragraph_attention = nn.Linear(
            self.paragraph_gru_out_size, self.paragraph_gru_out_size
        ).to(self.device)
        self.paragraph_context_layer = nn.Linear(
            self.paragraph_gru_out_size, 1, bias=False
        ).to(self.device)

        self.levels = LEVELS[levels]

        self.classifier_input_dim = (
            2 * paragraph_gru_hidden_size * 4 * len(self.levels)
        )  # 4 = number of concatenations

        self.classifier_output_dim = 1

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, self.classifier_input_dim),
            nn.ReLU(),
            nn.Linear(self.classifier_input_dim, self.classifier_output_dim),
        ).to(self.device)

    def forward(
        self, target_article, source_article, paragraphs_limit=None,
    ):
        target_article_representation = self.get_document_representation(
            target_article, levels=self.levels,
        )

        del target_article

        source_article_representation = self.get_document_representation(
            source_article, levels=self.levels
        )

        del source_article

        # Concatenates document representations. This is the siamese part of the model
        concatenated_articles_representation = torch.cat(
            (
                source_article_representation,
                target_article_representation,
                torch.abs(
                    source_article_representation - target_article_representation
                ),
                source_article_representation * target_article_representation,
            ),
            1,
        )

        del source_article_representation
        del target_article_representation

        articles_similarity = self.classifier(concatenated_articles_representation)

        return articles_similarity

    def get_document_representation(
        self, articles_batch, levels=None,
    ):
        document_representation_list = []
        for level in levels:
            (
                articles,
                paragraphs_per_document,
                sentences_per_paragraph,
                words_per_sentence,
            ) = self.get_tensors_at_level(articles_batch, level)

            batch_size = articles.shape[0]
            max_paragraphs_per_article = articles.shape[1]
            max_sentences_per_paragraph = articles.shape[2]
            max_words_per_sentence = words_per_sentence.max()

            flatten_size = (
                batch_size * max_paragraphs_per_article * max_sentences_per_paragraph
            )
            flatten_word_ids = articles.reshape((flatten_size, max_words_per_sentence))

            flatten_words_per_sentence = words_per_sentence.reshape(flatten_size)

            # deleting tensors to release memory
            del articles
            del words_per_sentence

            # Limit of number of elements per tensor
            TENSOR_SIZE_LIMIT = 500000

            split_factor = max(
                1, int(np.prod(flatten_word_ids.shape) / TENSOR_SIZE_LIMIT)
            )
            split_size = int(len(flatten_word_ids) / split_factor)
            flatten_word_ids = flatten_word_ids.split(split_size)
            flatten_words_per_sentence = flatten_words_per_sentence.split(split_size)

            word_level_representation_list = []
            for index, _ in enumerate(flatten_word_ids):
                word_level_representation = self.get_word_level_representation(
                    batch_size,
                    flatten_word_ids[index],
                    flatten_words_per_sentence[index],
                    max_paragraphs_per_article,
                    max_sentences_per_paragraph,
                    max_words_per_sentence,
                )

                word_level_representation_list.append(word_level_representation)

            word_level_representation = torch.cat(word_level_representation_list)

            word_level_representation = word_level_representation.reshape(
                (
                    batch_size * max_paragraphs_per_article,
                    max_sentences_per_paragraph,
                    word_level_representation.shape[-1],
                )
            )

            del flatten_word_ids
            del flatten_words_per_sentence

            # attention over sentences
            sentence_level_representation = self.get_sentence_level_representation(
                batch_size,
                word_level_representation,
                max_paragraphs_per_article,
                max_sentences_per_paragraph,
                sentences_per_paragraph,
            )

            # attention over paragraphs
            paragraph_level_representation = self.get_paragraph_level_representation(
                paragraphs_per_document, sentence_level_representation
            )

            document_representation_list.append(paragraph_level_representation)

        document_representation = torch.cat(document_representation_list, dim=1)

        return document_representation

    def get_tensors_at_level(self, articles_batch, level):
        if level == "sentence":
            articles = self.transform_to_sentence_level(articles_batch)

        elif level == "word":
            articles = self.transform_to_word_level(articles_batch)

        else:
            articles = articles_batch

        articles_text = articles[TEXT_IDS_COLUMN]
        words_per_sentence = articles[WORDS_PER_SENTENCE_COLUMN]
        sentences_per_paragraph = articles[SENTENCES_PER_PARAGRAPH_COLUMN]
        paragraphs_per_document = articles[PARAGRAPHS_PER_DOCUMENT_COLUMN]
        return (
            articles_text,
            paragraphs_per_document,
            sentences_per_paragraph,
            words_per_sentence,
        )

    def get_paragraph_level_representation(
        self, paragraphs_per_article, sentence_level_representation
    ):
        packed_paragraphs = pack_padded_sequence(
            sentence_level_representation,
            lengths=paragraphs_per_article,
            batch_first=True,
            enforce_sorted=False,
        )

        paragraph_gru_output, _ = self.paragraph_gru(packed_paragraphs)

        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        paragraph_state = torch.tanh(
            self.paragraph_attention(paragraph_gru_output.data)
        )

        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        paragraph_context_vector = self.paragraph_context_layer(paragraph_state)

        paragraph_attention = paragraph_state * paragraph_context_vector

        softmax_function = Softmax(1)

        paragraph_alphas = softmax_function(paragraph_attention)

        paragraph_alphas, _ = pad_packed_sequence(
            PackedSequence(
                data=paragraph_alphas,
                batch_sizes=paragraph_gru_output.batch_sizes,
                sorted_indices=paragraph_gru_output.sorted_indices,
                unsorted_indices=paragraph_gru_output.unsorted_indices,
            ),
            batch_first=True,
        )  # (n_sentences, max(words_per_sentence))

        paragraph_gru_output, _ = pad_packed_sequence(
            paragraph_gru_output, batch_first=True
        )

        document_representation = (paragraph_gru_output.float() * paragraph_alphas).sum(
            dim=1
        )

        return document_representation  # , paragraph_alphas

    def get_sentence_level_representation(
        self,
        batch_size,
        flatten_sentences,
        max_paragraphs_per_article,
        max_sentences_per_paragraph,
        sentences_per_paragraph,
    ):
        sentences_per_paragraph = sentences_per_paragraph.reshape(
            batch_size * max_paragraphs_per_article
        )
        non_empty_sentences_per_paragraph = remove_zeros(sentences_per_paragraph)

        packed_sentences = pack_padded_sequence(
            flatten_sentences,
            lengths=non_empty_sentences_per_paragraph,
            batch_first=True,
            enforce_sorted=False,
        )

        sentence_level_gru_output, _ = self.sentence_gru(packed_sentences)
        sentence_alphas = self.get_sentence_alphas(sentence_level_gru_output)

        sentence_level_gru_output, _ = pad_packed_sequence(
            sentence_level_gru_output, batch_first=True
        )

        try:
            sentence_level_representation = (
                sentence_alphas * sentence_level_gru_output
            ).sum(dim=1)

            sentence_level_representation = sentence_level_representation.reshape(
                batch_size,
                max_paragraphs_per_article,
                sentence_level_representation.shape[-1],
            )

        except:
            print("check")
            raise Exception("error")

        return sentence_level_representation  # , sentence_level_importance

    def get_word_level_representation(
        self,
        batch_size,
        flatten_word_ids,
        flatten_words_per_sentence,
        max_paragraphs_per_article,
        max_sentences_per_paragraph,
        max_words_per_sentence,
    ):
        word_embeddings = self.embedding(flatten_word_ids)
        # Cleaning memory
        del flatten_word_ids

        flatten_words_per_sentence = remove_zeros(flatten_words_per_sentence)

        packed_word_embeddings = pack_padded_sequence(
            word_embeddings,
            lengths=flatten_words_per_sentence,
            batch_first=True,
            enforce_sorted=False,
        ).float()
        # Cleaning memory
        del word_embeddings

        word_level_gru_output, _ = self.word_gru(packed_word_embeddings)
        # Cleaning memory
        del packed_word_embeddings

        word_alphas = self.get_word_alphas(word_level_gru_output)

        word_alphas, _ = pad_packed_sequence(
            PackedSequence(
                data=word_alphas,
                batch_sizes=word_level_gru_output.batch_sizes,
                sorted_indices=word_level_gru_output.sorted_indices,
                unsorted_indices=word_level_gru_output.unsorted_indices,
            ),
            batch_first=True,
        )

        # To track when memory limit is reached
        try:
            word_level_gru_output, _ = pad_packed_sequence(
                word_level_gru_output, batch_first=True
            )
        except Exception as error:
            print(
                f"Batch size: {batch_size} \n"
                f"Paragraphs per article: {max_paragraphs_per_article} \n"
                f"Sentences per paragraph: {max_sentences_per_paragraph} \n"
                f"Words per sentence: {max_words_per_sentence} "
            )

            logger.exception(error)

            exit(1)

        word_level_representation = (word_alphas * word_level_gru_output.float()).sum(
            dim=1
        )

        return word_level_representation

    def get_sentence_alphas(self, sentence_level_gru):
        sentence_state = torch.tanh(self.sentence_attention(sentence_level_gru.data))

        sentence_context_vector = self.sentence_context_layer(sentence_state)

        sentence_attention = sentence_state * sentence_context_vector

        softmax_function = Softmax(1)

        sentence_alphas = softmax_function(sentence_attention)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentence_alphas, _ = pad_packed_sequence(
            PackedSequence(
                data=sentence_alphas,
                batch_sizes=sentence_level_gru.batch_sizes,
                sorted_indices=sentence_level_gru.sorted_indices,
                unsorted_indices=sentence_level_gru.unsorted_indices,
            ),
            batch_first=True,
        )

        return sentence_alphas

    def get_word_alphas(self, word_gru_out):
        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        # This equation is represented in the paper as `u_{it}`
        word_state = torch.tanh(self.word_attention(word_gru_out.data))

        word_context_vector = self.word_context_layer(word_state)  # (n_words)

        word_attention = word_state * word_context_vector

        # Parameter `1` is the dimension the Softmax will be applied to
        softmax_function = Softmax(1)

        word_alphas = softmax_function(word_attention)

        return word_alphas

    def transform_to_word_level(self, document):
        batch_size = document[TEXT_IDS_COLUMN].shape[0]

        document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_document_at_word_level(
            document[WORDS_PER_SENTENCE_COLUMN]
        )
        document[TEXT_IDS_COLUMN] = get_document_at_word_level(
            document[TEXT_IDS_COLUMN], document[WORDS_PER_SENTENCE_COLUMN], self.device
        )
        document[SENTENCES_PER_PARAGRAPH_COLUMN] = torch.ones(
            (batch_size, 1), dtype=int, device=self.device
        )
        document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(
            batch_size, dtype=int, device=self.device
        )

        return document

    def transform_to_sentence_level(self, document):
        batch_size = document[TEXT_IDS_COLUMN].shape[0]

        document[TEXT_IDS_COLUMN] = get_document_at_sentence_level(
            document[TEXT_IDS_COLUMN], self.device
        )
        document[WORDS_PER_SENTENCE_COLUMN] = get_words_per_sentence_at_sentence_level(
            document[WORDS_PER_SENTENCE_COLUMN], self.device
        )
        document[
            SENTENCES_PER_PARAGRAPH_COLUMN
        ] = get_sentences_per_paragraph_at_sentence_level(
            document[SENTENCES_PER_PARAGRAPH_COLUMN]
        )
        document[PARAGRAPHS_PER_DOCUMENT_COLUMN] = torch.ones(
            batch_size, dtype=int, device=self.device
        )

        return document


if __name__ == "__main__":
    word2vec_path = "../../data/source/glove.6B.50d.txt"
    dict = pd.read_csv(
        filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE
    ).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(
        np.concatenate([unknown_word, dict], axis=0).astype(np.float)
    )
    model = SmashRNNModel(dict, dict_len, embed_dim)
    # batch_size = 6
    # max_paragraphs_per_article = 111
    # max_sentences_per_paragraph = 22
    # max_words_per_sentence = 26
    # test_tensor = torch.zeros(
    #     (batch_size, max_paragraphs_per_article, max_sentences_per_paragraph, max_words_per_sentence), dtype=int
    # )
    #
    # print(test_tensor.shape)
    #
    # paragraphs_per_article = torch.ones(batch_size, dtype=int)
    # sentences_per_article = torch.zeros((batch_size, max_paragraphs_per_article), dtype=int)
    # words_per_sentence = torch.zeros((batch_size, max_paragraphs_per_article, max_sentences_per_paragraph), dtype=int)
    #
    # print(
    #     model.get_document_representation(
    #         test_tensor, paragraphs_per_article, sentences_per_article, words_per_sentence
    #     )
    # )
    print("a")
