"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import gc
import logging
import os
import sys

import psutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from utils.utils import remove_zeros

logger = logging.getLogger(__name__)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

GRU_HIDDEN_SIZE = 128
HIDDEN_LAYER_SIZE = 128


class SmashRNNModel(nn.Module):
    def __init__(self, dict, dict_len, embedding_size):
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
            nn.Embedding(num_embeddings=dict_len, embedding_dim=embedding_size).from_pretrained(dict).to(self.device)
        )

        # RNN + attention layers
        word_gru_hidden_size = GRU_HIDDEN_SIZE
        self.word_gru_out_size = word_gru_hidden_size * 2
        self.word_gru = nn.GRU(embedding_size, word_gru_hidden_size, bidirectional=True, batch_first=True).to(
            self.device
        )
        self.word_attention = nn.Linear(self.word_gru_out_size, HIDDEN_LAYER_SIZE).to(self.device)
        self.word_context_vector = nn.Linear(HIDDEN_LAYER_SIZE, 1, bias=False).to(
            self.device
        )  # Word context vector to take dot-product with

        sentence_gru_hidden_size = GRU_HIDDEN_SIZE
        self.sentence_gru_out_size = sentence_gru_hidden_size * 2
        self.sentence_gru = nn.GRU(
            self.word_gru_out_size, sentence_gru_hidden_size, bidirectional=True, batch_first=True
        ).to(self.device)
        self.sentence_attention = nn.Linear(self.sentence_gru_out_size, HIDDEN_LAYER_SIZE).to(self.device)
        self.sentence_context_vector = nn.Linear(HIDDEN_LAYER_SIZE, 1, bias=False).to(self.device)

        paragraph_gru_hidden_size = GRU_HIDDEN_SIZE
        self.paragraph_gru_out_size = paragraph_gru_hidden_size * 2
        self.paragraph_gru = nn.GRU(
            self.sentence_gru_out_size, paragraph_gru_hidden_size, bidirectional=True, batch_first=True
        ).to(self.device)
        self.paragraph_attention = nn.Linear(self.paragraph_gru_out_size, HIDDEN_LAYER_SIZE).to(self.device)
        self.paragraph_context_vector = nn.Linear(HIDDEN_LAYER_SIZE, 1, bias=False).to(self.device)

        self.input_dim = 2 * paragraph_gru_hidden_size * 4  # 4 = number of concatenations

        # Not mentioned in the paper.
        self.mlp_dim = int(self.input_dim / 2)
        self.out_dim = 1

        # These layers compute the semantic similarity between two documents
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.out_dim), nn.Sigmoid()
        ).to(self.device)

    def forward(
        self,
        target_article,
        words_per_sentence_target_article,
        sentences_per_paragraph_target_article,
        paragraphs_per_article_target_article,
        source_article,
        words_per_sentence_source_article,
        sentences_per_paragraph_source_article,
        paragraphs_per_article_source_article,
        paragraphs_limit=None,
    ):

        target_article_representation = self.get_document_representation(
            target_article,
            paragraphs_per_article_target_article,
            sentences_per_paragraph_target_article,
            words_per_sentence_target_article,
            paragraphs_limit,
        )

        del target_article
        del paragraphs_per_article_target_article
        del sentences_per_paragraph_target_article
        del words_per_sentence_target_article

        source_article_representation = self.get_document_representation(
            source_article,
            paragraphs_per_article_source_article,
            sentences_per_paragraph_source_article,
            words_per_sentence_source_article,
            paragraphs_limit,
        )

        del source_article
        del paragraphs_per_article_source_article
        del sentences_per_paragraph_source_article
        del words_per_sentence_source_article

        # Concatenates document representations. This is the siamese part of the model
        concatenated_articles_representation = torch.cat(
            (
                source_article_representation,
                target_article_representation,
                torch.abs(source_article_representation - target_article_representation),
                source_article_representation * target_article_representation,
            ),
            1,
        )

        del source_article_representation
        del target_article_representation

        articles_similarity = self.classifier(concatenated_articles_representation)

        return articles_similarity

    def get_document_representation(
        self, articles_batch, paragraphs_per_article, sentences_per_paragraph, words_per_sentence, paragraphs_limit=None
    ):

        words_per_sentence_limit = 50

        # TODO this should be in the dataset class
        if paragraphs_limit and articles_batch.shape[1] > paragraphs_limit:
            paragraphs_per_article = paragraphs_per_article.clamp(1, paragraphs_limit)
            sentences_per_paragraph = sentences_per_paragraph[:, :paragraphs_limit]
            max_sentences_per_paragraph = sentences_per_paragraph.max()

            words_per_sentence = words_per_sentence[:, :paragraphs_limit, :max_sentences_per_paragraph]
            max_words_per_sentence = words_per_sentence.max()

            articles_batch = articles_batch[:, :paragraphs_limit, :max_sentences_per_paragraph, :max_words_per_sentence]

        # if articles_batch.shape[3] > words_per_sentence_limit:
        #     words_per_sentence = words_per_sentence.clamp(1, words_per_sentence_limit)
        #     articles_batch = articles_batch[:, :, :, :words_per_sentence_limit]

        batch_size = articles_batch.shape[0]
        max_paragraphs_per_article = articles_batch.shape[1]
        max_sentences_per_paragraph = articles_batch.shape[2]
        max_words_per_sentence = words_per_sentence.max()

        flatten_size = batch_size * max_paragraphs_per_article * max_sentences_per_paragraph
        flatten_word_ids = articles_batch.reshape((flatten_size, max_words_per_sentence))

        flatten_words_per_sentence = words_per_sentence.reshape(flatten_size)

        # deleting tensors to release memory
        del articles_batch
        del words_per_sentence

        # Limit of number of elements per tensor
        TENSOR_SIZE_LIMIT = 1000000

        split_factor = max(1, int(np.prod(flatten_word_ids.shape) / TENSOR_SIZE_LIMIT))
        split_size = int(len(flatten_word_ids) / split_factor)
        flatten_word_ids = flatten_word_ids.split(split_size)
        flatten_words_per_sentence = flatten_words_per_sentence.split(split_size)

        word_level_importance_list = []
        word_level_representation_list = []
        for index, _ in enumerate(flatten_word_ids):
            # attention over words
            word_level_importance, word_level_representation = self.get_word_level_representation(
                batch_size,
                flatten_word_ids[index],
                flatten_words_per_sentence[index],
                max_paragraphs_per_article,
                max_sentences_per_paragraph,
                max_words_per_sentence,
            )

            word_level_representation_list.append(word_level_representation)
            word_level_importance_list.append(word_level_importance)

        word_level_representation = torch.cat(word_level_representation_list)
        word_level_importance = torch.cat(word_level_importance_list)

        word_level_representation = word_level_representation.reshape(
            (batch_size * max_paragraphs_per_article, max_sentences_per_paragraph, word_level_representation.shape[-1])
        )
        word_level_importance = word_level_importance.reshape(
            (batch_size * max_paragraphs_per_article, max_sentences_per_paragraph, max_words_per_sentence)
        )

        del flatten_word_ids
        del flatten_words_per_sentence

        # attention over sentences
        sentence_level_importance, sentence_level_representation = self.get_sentence_level_representation(
            batch_size,
            word_level_representation,
            max_paragraphs_per_article,
            max_sentences_per_paragraph,
            sentences_per_paragraph,
        )

        del sentences_per_paragraph

        # attention over paragraphs
        paragraph_alphas, document_representation = self.get_paragraph_level_representation(
            paragraphs_per_article, sentence_level_representation
        )
        # (batch_size, self.paragraph_gru_out_size)

        # # Calculating most important words
        # # Gets the paragraph with max attention per document in batch
        # maximum_paragraph_indices = [
        #     (document == document.max()).nonzero().squeeze(1).item() for document in paragraph_alphas
        # ]
        #
        # # Gets the sentence with max attention per sentence in the max paragraph in document in batch
        # maximum_sentence = []
        # for document_index, document_max_paragraph in enumerate(maximum_paragraph_indices):
        #     maximum_sentence.append(sentence_level_importance[document_index][document_max_paragraph])
        #
        # maximum_sentence_indices = [
        #     (document == document.max()).nonzero().squeeze(1).item() for document in maximum_sentence
        # ]
        #
        # # Gets the word with max attention in the sentence with max attention in the paragraph with max attention
        # # per document in batch
        # maximum_words_indices = []
        # for document_index, document_max_sentence in enumerate(maximum_sentence_indices):
        #     document_max_paragraph = maximum_paragraph_indices[document_index]
        #     maximum_words_indices.append(
        #         word_level_importance[document_index, document_max_paragraph, document_max_sentence]
        #     )
        #
        # maximum_word_indices = [
        #     (document == document.max()).nonzero().squeeze(1).item() for document in maximum_words_indices
        # ]

        return document_representation

    def get_paragraph_level_representation(self, paragraphs_per_article, sentence_level_representation):
        packed_paragraphs = pack_padded_sequence(
            sentence_level_representation, lengths=paragraphs_per_article, batch_first=True, enforce_sorted=False
        )
        paragraph_gru_out, _ = self.paragraph_gru(packed_paragraphs)
        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        paragraph_att_out = torch.tanh(self.paragraph_attention(paragraph_gru_out.data))
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        paragraph_att_out = self.paragraph_context_vector(paragraph_att_out).squeeze(1)  # (n_words)
        max_value = paragraph_att_out.max()
        paragraph_att_out = torch.exp(paragraph_att_out - max_value)  # (n_words)
        paragraph_att_out, _ = pad_packed_sequence(
            PackedSequence(
                data=paragraph_att_out,
                batch_sizes=paragraph_gru_out.batch_sizes,
                sorted_indices=paragraph_gru_out.sorted_indices,
                unsorted_indices=paragraph_gru_out.unsorted_indices,
            ),
            batch_first=True,
        )  # (n_sentences, max(words_per_sentence))
        paragraph_alphas = paragraph_att_out / torch.sum(paragraph_att_out, dim=1, keepdim=True)
        # (n_sentences, max(words_per_sentence))
        paragraph_gru_out, _ = pad_packed_sequence(paragraph_gru_out, batch_first=True)
        # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        document_representation = (paragraph_gru_out.float() * paragraph_alphas.unsqueeze(2)).sum(dim=1)
        return paragraph_alphas, document_representation

    def get_sentence_level_representation(
        self,
        batch_size,
        flatten_sentences,
        max_paragraphs_per_article,
        max_sentences_per_paragraph,
        sentences_per_paragraph,
    ):
        sentences_per_paragraph = sentences_per_paragraph.reshape(batch_size * max_paragraphs_per_article)
        non_empty_sentences_per_paragraph = remove_zeros(sentences_per_paragraph)

        packed_sentences = pack_padded_sequence(
            flatten_sentences, lengths=non_empty_sentences_per_paragraph, batch_first=True, enforce_sorted=False,
        )

        sentence_level_gru, _ = self.sentence_gru(packed_sentences)
        sentence_level_attention = self.get_sentence_level_attention(sentence_level_gru)
        sentence_alphas = self.get_alphas(sentence_level_attention)

        sentence_level_importance = sentence_alphas.reshape(
            (batch_size, max_paragraphs_per_article, sentences_per_paragraph.max())
        )

        # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
        sentence_level_gru, _ = pad_packed_sequence(sentence_level_gru, batch_first=True)
        sentence_level_representation = self.get_representation(sentence_alphas, sentence_level_gru).reshape(
            (batch_size, max_paragraphs_per_article, sentence_level_gru.shape[-1])
        )

        return sentence_level_importance, sentence_level_representation

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
        flatten_words_per_sentence = remove_zeros(flatten_words_per_sentence)

        packed_word_embeddings = pack_padded_sequence(
            word_embeddings, lengths=flatten_words_per_sentence, batch_first=True, enforce_sorted=False
        ).float()

        word_level_gru, _ = self.word_gru(packed_word_embeddings)
        word_level_attention = self.get_word_attention(word_level_gru)

        word_level_attention, _ = pad_packed_sequence(
            PackedSequence(
                data=word_level_attention,
                batch_sizes=word_level_gru.batch_sizes,
                sorted_indices=word_level_gru.sorted_indices,
                unsorted_indices=word_level_gru.unsorted_indices,
            ),
            batch_first=True,
        )

        word_level_alphas = self.get_alphas(word_level_attention)

        # Clean memory here
        del flatten_word_ids
        del packed_word_embeddings
        del word_embeddings
        del word_level_attention

        # To track when memory limit is reached
        try:
            word_level_gru, _ = pad_packed_sequence(word_level_gru, batch_first=True)
        except Exception as error:
            print(
                f"Batch size: {batch_size} \n"
                f"Paragraphs per article: {max_paragraphs_per_article} \n"
                f"Sentences per paragraph: {max_sentences_per_paragraph} \n"
                f"Words per sentence: {max_words_per_sentence} "
            )

            logger.exception(error)

            exit(1)

        word_level_representation = self.get_representation(word_level_alphas, word_level_gru)

        return word_level_alphas, word_level_representation

    def get_sentence_level_attention(self, sentence_level_gru):
        sentence_level_attention = torch.tanh(self.sentence_attention(sentence_level_gru.data))
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        sentence_level_attention = self.sentence_context_vector(sentence_level_attention).squeeze(1)  # (n_words)
        sentence_level_attention = self.softmax(sentence_level_attention)
        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentence_level_attention, _ = pad_packed_sequence(
            PackedSequence(
                data=sentence_level_attention,
                batch_sizes=sentence_level_gru.batch_sizes,
                sorted_indices=sentence_level_gru.sorted_indices,
                unsorted_indices=sentence_level_gru.unsorted_indices,
            ),
            batch_first=True,
        )  # (n_sentences, max(words_per_sentence))
        return sentence_level_attention

    @staticmethod
    def softmax(sentence_level_attention):
        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence
        # First, take the exponent
        max_value = sentence_level_attention.max()  # scalar, for numerical stability during exponent calculation
        sentence_level_attention = torch.exp(sentence_level_attention - max_value)  # (n_words)
        return sentence_level_attention

    @staticmethod
    def get_representation(alphas, gru_output):
        # gets the representation for the sentence
        sentence_representation = (alphas * gru_output.float()).sum(dim=1)  # (batch_size, gru_out_size)

        return sentence_representation

    def get_word_attention(self, word_gru_out):
        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        word_att_out = torch.tanh(self.word_attention(word_gru_out.data))
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        word_att_out = self.word_context_vector(word_att_out).squeeze(1)  # (n_words)
        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence
        # First, take the exponent
        max_value = word_att_out.max()  # scalar, for numerical stability during exponent calculation
        word_att_out = torch.exp(word_att_out - max_value)  # (n_words)

        return word_att_out

    def get_word_gru(self, word_ids, words_per_sentence_in_paragraph):
        # get word embeddings from ids
        word_embeddings = self.embedding(word_ids)
        packed_word_embeddings = pack_padded_sequence(
            word_embeddings, lengths=words_per_sentence_in_paragraph, batch_first=True, enforce_sorted=False
        ).float()

        word_level_gru, _ = self.word_gru(packed_word_embeddings)

        return word_level_gru

    @staticmethod
    def get_alphas(attention_representation):
        alphas = attention_representation / torch.sum(attention_representation, dim=1, keepdim=True)
        alphas = torch.where(torch.isnan(alphas), torch.zeros_like(alphas), alphas)

        return alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence))


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def cpuStats():
    # print(sys.version)
    # print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB...I think
    print("memory GB:", memoryUse)


if __name__ == "__main__":
    word2vec_path = "../../data/source/glove.6B.50d.txt"
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
    model = SmashRNNModel(dict, dict_len, embed_dim)
    batch_size = 6
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
