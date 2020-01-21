"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import torch.nn.functional as F

from src.utils import get_document_at_word_level


class WordLevelSmashRNNModel(nn.Module):
    def __init__(self, dict, dict_len, embedding_size, max_word_length, max_sent_length, max_paragraph_length):
        super(WordLevelSmashRNNModel, self).__init__()

        self.max_word_length = 500
        self.batch_size = 1

        # Init embedding layer
        self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embedding_size).from_pretrained(dict)

        # RNN + attention layers
        word_gru_hidden_size = 100
        self.word_gru_out_size = word_gru_hidden_size * 2
        self.word_gru = nn.GRU(embedding_size, word_gru_hidden_size, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(self.word_gru_out_size, 50)
        self.word_context_vector = nn.Linear(50, 1, bias=False)  # Word context vector to take dot-product with

        self.input_dim = 2 * word_gru_hidden_size * 3  # 3 = number of concatenations

        # Not mentioned in the paper.
        self.mlp_dim = 200  # int(self.input_dim / 2)
        self.out_dim = 1

        # These layers compute the semantic similarity between two documents
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, current_document, words_per_sentence_current_document,
                previous_document, words_per_sentence_previous_document,
                click_rate_tensor):
        current_document = get_document_at_word_level(current_document, words_per_sentence_current_document)
        previous_document = get_document_at_word_level(previous_document, words_per_sentence_previous_document)

        # This only works with self.batch_size = 1
        current_document_representation = self.get_document_representation(current_document)
        previous_document_representation = self.get_document_representation(previous_document)

        # Concatenates document representations. This is the siamese part of the model
        concatenated_documents_representation = torch.cat((current_document_representation,
                                                           previous_document_representation,
                                                           torch.abs(
                                                               current_document_representation - previous_document_representation),
                                                           ), 1)

        predicted_ctr = self.classifier(concatenated_documents_representation)

        return predicted_ctr

    def get_document_representation(self, word_ids_in_sent):
        if torch.cuda.is_available():
            word_ids_in_sent = word_ids_in_sent.cuda()

        # attention over words
        # 1st dim = batch, last dim = words
        # get word embeddings from ids
        words_in_sent = self.embedding(word_ids_in_sent)

        word_gru_out, _ = self.word_gru(words_in_sent.float())

        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        word_att_out = torch.tanh(self.word_attention(word_gru_out.data))

        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        word_att_out = self.word_context_vector(word_att_out).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = word_att_out.max()  # scalar, for numerical stability during exponent calculation
        word_att_out = torch.exp(word_att_out - max_value)  # (n_words)

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = word_att_out / torch.sum(word_att_out, dim=1,
                                               keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Find sentence embeddings
        # gets the representation for the sentence
        sentence_representation = (word_gru_out.float() * word_alphas).sum(
            dim=1)  # (batch_size, self.word_gru_out_size)

        return sentence_representation


if __name__ == '__main__':
    word2vec_path = '../data/glove.6B.50d.txt'
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
    WordLevelSmashRNNModel(dict, dict_len, embed_dim, 38, 60, 90)
