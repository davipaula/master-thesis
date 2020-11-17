"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

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

        self.levels = LEVELS[levels]

        # Init embedding layer
        self.embedding = (
            nn.Embedding(num_embeddings=dict_len, embedding_dim=embedding_size)
            .from_pretrained(dict)
            .to(self.device)
        )

        # RNN + attention layers
        self.word_gru = nn.GRU(
            embedding_size, GRU_HIDDEN_SIZE, bidirectional=True, batch_first=True
        ).to(self.device)

        self.word_gru_out_size = GRU_HIDDEN_SIZE * 2

        self.word_attention = nn.Linear(self.word_gru_out_size, GRU_HIDDEN_SIZE).to(
            self.device
        )
        self.word_context_layer = nn.Linear(GRU_HIDDEN_SIZE, 1, bias=False).to(
            self.device
        )

        sentence_gru_hidden_size = GRU_HIDDEN_SIZE
        self.sentence_gru_out_size = sentence_gru_hidden_size * 2
        self.sentence_gru = nn.GRU(
            self.word_gru_out_size,
            sentence_gru_hidden_size,
            bidirectional=True,
            batch_first=True,
        ).to(self.device)
        self.sentence_attention = nn.Linear(
            self.sentence_gru_out_size, sentence_gru_hidden_size
        ).to(self.device)
        self.sentence_context_layer = nn.Linear(
            sentence_gru_hidden_size, 1, bias=False
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
            self.paragraph_gru_out_size, GRU_HIDDEN_SIZE
        ).to(self.device)
        self.paragraph_context_layer = nn.Linear(GRU_HIDDEN_SIZE, 1, bias=False).to(
            self.device
        )

        # Estimator config
        self.estimator_input_dim = (
            2 * GRU_HIDDEN_SIZE * 4 * len(self.levels)
        )  # 4 = number of concatenations

        self.estimator_output_dim = 1

        self.estimator = nn.Sequential(
            nn.Linear(self.estimator_input_dim, self.estimator_input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.estimator_input_dim // 2, self.estimator_output_dim),
        ).to(self.device)

    def forward(
        self, target_article, source_article, paragraphs_limit=None,
    ):
        target_article_representation = self.get_document_representation(
            target_article, levels=self.levels, paragraphs_limit=paragraphs_limit
        )

        del target_article

        source_article_representation = self.get_document_representation(
            source_article, levels=self.levels, paragraphs_limit=paragraphs_limit
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

        articles_similarity = self.estimator(concatenated_articles_representation)

        return articles_similarity

    def get_document_representation(
        self, articles_batch, levels=None, paragraphs_limit=None
    ):
        document_representation_list = []
        for level in levels:
            (
                articles,
                paragraphs_per_document,
                sentences_per_paragraph,
                words_per_sentence,
            ) = self.get_tensors_at_level(articles_batch, level)

            if paragraphs_limit and articles.shape[1] > paragraphs_limit:
                paragraphs_per_document = paragraphs_per_document.clamp(
                    1, paragraphs_limit
                )
                sentences_per_paragraph = sentences_per_paragraph[:, :paragraphs_limit]
                max_sentences_per_paragraph = sentences_per_paragraph.max()

                words_per_sentence = words_per_sentence[
                    :, :paragraphs_limit, :max_sentences_per_paragraph
                ]
                max_words_per_sentence = words_per_sentence.max()

                articles = articles[
                    :,
                    :paragraphs_limit,
                    :max_sentences_per_paragraph,
                    :max_words_per_sentence,
                ]

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

        articles_text = articles[TEXT_IDS_COLUMN].to(self.device, non_blocking=True)
        words_per_sentence = articles[WORDS_PER_SENTENCE_COLUMN].to(
            self.device, non_blocking=True
        )
        sentences_per_paragraph = articles[SENTENCES_PER_PARAGRAPH_COLUMN].to(
            self.device, non_blocking=True
        )
        paragraphs_per_document = articles[PARAGRAPHS_PER_DOCUMENT_COLUMN].to(
            self.device, non_blocking=True
        )
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
        paragraph_context_vector = self.paragraph_context_layer(
            paragraph_state
        ).squeeze(1)

        exponent = torch.exp(paragraph_context_vector)

        del paragraph_state
        del paragraph_context_vector

        paragraph_attention, _ = pad_packed_sequence(
            PackedSequence(
                data=exponent,
                batch_sizes=paragraph_gru_output.batch_sizes,
                sorted_indices=paragraph_gru_output.sorted_indices,
                unsorted_indices=paragraph_gru_output.unsorted_indices,
            ),
            batch_first=True,
        )  # (n_sentences, max(words_per_sentence))

        paragraph_alphas = paragraph_attention / torch.sum(
            paragraph_attention, dim=1, keepdim=True
        )

        paragraph_gru_output, _ = pad_packed_sequence(
            paragraph_gru_output, batch_first=True
        )

        document_representation = (
            paragraph_gru_output * paragraph_alphas.unsqueeze(2)
        ).sum(dim=1)

        return document_representation  # , paragraph_alphas

    def get_sentence_level_representation(
        self,
        batch_size,
        flatten_sentences,
        max_paragraphs_per_article,
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
        del flatten_sentences

        sentence_level_gru_output, _ = self.sentence_gru(packed_sentences)
        sentence_alphas = self.get_sentence_alphas(sentence_level_gru_output)

        sentence_level_gru_output, _ = pad_packed_sequence(
            sentence_level_gru_output, batch_first=True
        )

        sentence_level_representation = (
            sentence_alphas.unsqueeze(2) * sentence_level_gru_output
        ).sum(dim=1)

        sentence_level_representation = sentence_level_representation.reshape(
            batch_size,
            max_paragraphs_per_article,
            sentence_level_representation.shape[-1],
        )

        return sentence_level_representation

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
        del flatten_words_per_sentence

        word_level_gru_output, _ = self.word_gru(packed_word_embeddings)
        # Cleaning memory
        del packed_word_embeddings

        word_alphas = self.get_word_alphas(word_level_gru_output)

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

        word_level_representation = (
            word_alphas.unsqueeze(2) * word_level_gru_output
        ).sum(dim=1)

        return word_level_representation

    def get_sentence_alphas(self, sentence_level_gru):
        sentence_state = torch.tanh(self.sentence_attention(sentence_level_gru.data))

        sentence_context_vector = self.sentence_context_layer(sentence_state).squeeze(1)

        exponent = torch.exp(sentence_context_vector)

        del sentence_context_vector
        del sentence_state

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentence_attention, _ = pad_packed_sequence(
            PackedSequence(
                data=exponent,
                batch_sizes=sentence_level_gru.batch_sizes,
                sorted_indices=sentence_level_gru.sorted_indices,
                unsorted_indices=sentence_level_gru.unsorted_indices,
            ),
            batch_first=True,
        )

        sentence_alphas = sentence_attention / torch.sum(
            sentence_attention, dim=1, keepdim=True
        )
        del sentence_attention

        return sentence_alphas

    def get_word_alphas(self, word_gru_out):
        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        # This equation is represented in the paper as `u_{it}`
        word_state = torch.tanh(self.word_attention(word_gru_out.data))

        word_context_vector = self.word_context_layer(word_state).squeeze(1)
        # (n_words)

        exponent = torch.exp(word_context_vector)

        del word_state
        del word_context_vector

        word_attention, _ = pad_packed_sequence(
            PackedSequence(
                data=exponent,
                batch_sizes=word_gru_out.batch_sizes,
                sorted_indices=word_gru_out.sorted_indices,
                unsorted_indices=word_gru_out.unsorted_indices,
            ),
            batch_first=True,
        )

        word_alphas = word_attention / torch.sum(word_attention, dim=1, keepdim=True)

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
