"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import csv
import os
from ast import literal_eval
import gensim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import DataLoader, TensorDataset
from utils import get_max_lengths
from smash_rnn_model import SmashRNNModel
from tensorboardX import SummaryWriter
import argparse
import shutil
from json_dataset import SMASHDataset


class SmashRNN:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        self.batch_size = 1

        train_dataset_path = './data/wiki_df_small.csv'
        word2vec_path = './data/glove.6B.50d.txt'

        self.max_word_length, self.max_sent_length, self.max_paragraph_length = get_max_lengths(train_dataset_path)

        self.training_dataset = SMASHDataset(train_dataset_path, word2vec_path, self.max_sent_length,
                                             self.max_word_length,
                                             self.max_paragraph_length)

        words_ids_current_document = [literal_eval(text) for text in self.training_dataset.current_article_text.values]
        words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document = self.get_padded_document_structures(
            words_ids_current_document)
        current_document_tensor = torch.LongTensor(
            [self.training_dataset.get_padded_document(current_document) for current_document in
             words_ids_current_document])

        words_ids_previous_document = [literal_eval(text) for text in
                                       self.training_dataset.previous_article_text.values]
        words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document = self.get_padded_document_structures(
            words_ids_previous_document)
        previous_document_tensor = torch.LongTensor(
            [self.training_dataset.get_padded_document(previous_document) for previous_document in
             words_ids_previous_document])

        click_rate_tensor = torch.Tensor([click_rate for click_rate in self.training_dataset.click_rate])

        self.dataset = TensorDataset(current_document_tensor, words_per_sentence_current_document,
                                     sentences_per_paragraph_current_document, paragraphs_per_document_current_document,
                                     previous_document_tensor, words_per_sentence_previous_document,
                                     sentences_per_paragraph_previous_document,
                                     paragraphs_per_document_previous_document,
                                     click_rate_tensor)

        # Load from txt file (in word2vec format)
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_dim = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_dim))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        # Convert to PyTorch tensor
        # weights = torch.FloatTensor(w2v_model.vectors)

        # Init embedding layer
        self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_dim).from_pretrained(dict)

        # RNN + attention layers
        word_gru_hidden_size = 100
        self.word_gru_out_size = word_gru_hidden_size * 2
        self.word_gru = nn.GRU(embed_dim, word_gru_hidden_size, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(self.word_gru_out_size, 50)
        self.word_context_vector = nn.Linear(50, 1, bias=False)  # Word context vector to take dot-product with

        sentence_gru_hidden_size = 200
        self.sentence_gru_out_size = sentence_gru_hidden_size * 2
        self.sentence_gru = nn.GRU(self.word_gru_out_size, sentence_gru_hidden_size, bidirectional=True,
                                   batch_first=True)
        self.sentence_attention = nn.Linear(self.sentence_gru_out_size, 50)
        self.sentence_context_vector = nn.Linear(50, 1, bias=False)

        paragraph_gru_hidden_size = 300
        self.paragraph_gru_out_size = paragraph_gru_hidden_size * 2
        self.paragraph_gru = nn.GRU(self.sentence_gru_out_size, paragraph_gru_hidden_size, bidirectional=True,
                                    batch_first=True)
        self.paragraph_attention = nn.Linear(self.paragraph_gru_out_size, 50)
        self.paragraph_context_vector = nn.Linear(50, 1, bias=False)

        self.input_dim = 2 * paragraph_gru_hidden_size * 3  # 3 = number of concatenations

        # Not mentioned in the paper.
        self.mlp_dim = int(self.input_dim / 2)
        self.out_dim = 1

        # These layers compute the semantic similarity between two documents
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, current_document, words_per_sentence_current_document,
                sentences_per_paragraph_current_document, paragraphs_per_document_current_document,
                previous_document, words_per_sentence_previous_document,
                sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document,
                click_rate_tensor):
        # This only works with self.batch_size = 1
        current_document_representation = self.get_document_representation(current_document,
                                                                           paragraphs_per_document_current_document,
                                                                           sentences_per_paragraph_current_document,
                                                                           words_per_sentence_current_document)
        previous_document_representation = self.get_document_representation(previous_document,
                                                                            paragraphs_per_document_previous_document,
                                                                            sentences_per_paragraph_previous_document,
                                                                            words_per_sentence_previous_document)

        # Concatenates document representations. This is the siamese part of the model
        concatenated_documents_representation = torch.cat((current_document_representation,
                                                           previous_document_representation,
                                                           torch.abs(
                                                               current_document_representation - previous_document_representation),
                                                           ), 1)

        predicted_ctr = self.classifier(concatenated_documents_representation)

        return predicted_ctr

    def get_document_representation(self, current_document, paragraphs_per_document_current_document,
                                    sentences_per_paragraph_current_document, words_per_sentence_current_document):

        # this only works with batch_size = 1
        _paragraphs_per_doc = paragraphs_per_document_current_document.item()
        _sentences_per_paragraph = [[words for words in paragraph if words > 0] for paragraph in
                                    sentences_per_paragraph_current_document.tolist()][0]
        _words_per_sentence = words_per_sentence_current_document
        # zero placeholders
        sentences = torch.zeros(
            (self.batch_size, self.max_paragraph_length, self.max_sent_length, self.word_gru_out_size))
        paragraphs = torch.zeros((self.batch_size, self.max_paragraph_length, self.sentence_gru_out_size))
        docs = torch.zeros((self.batch_size, self.paragraph_gru_out_size))
        # iterate over each hierarchy level
        for paragraph_idx in range(_paragraphs_per_doc):
            for sentence_idx in range(_sentences_per_paragraph[paragraph_idx]):
                # attention over words
                word_ids_in_sent = current_document[:, paragraph_idx, sentence_idx, :]
                # 1st dim = batch, last dim = words
                words_in_sent = self.embedding(word_ids_in_sent)  # get word embeddings from ids

                # pack padded sequence
                packed_words = pack_padded_sequence(words_in_sent,
                                                    lengths=torch.LongTensor(
                                                        words_per_sentence_current_document[:, paragraph_idx,
                                                        sentence_idx].tolist()),
                                                    batch_first=True, enforce_sorted=False)

                word_gru_out, _ = self.word_gru(packed_words.float())

                # This implementation uses the feature sentence_embeddings. Paper uses hidden state
                word_att_out = torch.tanh(self.word_attention(word_gru_out.data))

                # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
                word_att_out = self.word_context_vector(word_att_out).squeeze(1)  # (n_words)

                # Compute softmax over the dot-product manually
                # Manually because they have to be computed only over words in the same sentence

                # First, take the exponent
                max_value = word_att_out.max()  # scalar, for numerical stability during exponent calculation
                word_att_out = torch.exp(word_att_out - max_value)  # (n_words)

                # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
                word_att_out, _ = pad_packed_sequence(PackedSequence(data=word_att_out,
                                                                     batch_sizes=word_gru_out.batch_sizes,
                                                                     sorted_indices=word_gru_out.sorted_indices,
                                                                     unsorted_indices=word_gru_out.unsorted_indices),
                                                      batch_first=True)  # (n_sentences, max(words_per_sentence))

                # Calculate softmax values as now words are arranged in their respective sentences
                word_alphas = word_att_out / torch.sum(word_att_out, dim=1,
                                                       keepdim=True)  # (n_sentences, max(words_per_sentence))

                # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
                _sentence, _ = pad_packed_sequence(word_gru_out,
                                                   batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

                # Find sentence embeddings
                # gets the representation for the sentence
                _sentence = (_sentence.float() * word_alphas.unsqueeze(2)).sum(
                    dim=1)  # (batch_size, self.word_gru_out_size)

                sentences[:, paragraph_idx, sentence_idx] = _sentence

            # attention over sentences
            sentences_in_paragraph = sentences[:, paragraph_idx, :]

            # pack padded sequence of sentences
            packed_sentences = pack_padded_sequence(sentences_in_paragraph,
                                                    lengths=sentences_per_paragraph_current_document[:,
                                                            paragraph_idx].tolist(),
                                                    batch_first=True, enforce_sorted=False)

            sentence_gru_out, _ = self.sentence_gru(packed_sentences)

            sent_att_out = torch.tanh(self.sentence_attention(sentence_gru_out.data))

            # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
            sent_att_out = self.sentence_context_vector(sent_att_out).squeeze(1)  # (n_words)

            # Compute softmax over the dot-product manually
            # Manually because they have to be computed only over words in the same sentence

            # First, take the exponent
            max_value = sent_att_out.max()  # scalar, for numerical stability during exponent calculation
            sent_att_out = torch.exp(sent_att_out - max_value)  # (n_words)

            # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
            sent_att_out, _ = pad_packed_sequence(PackedSequence(data=sent_att_out,
                                                                 batch_sizes=sentence_gru_out.batch_sizes,
                                                                 sorted_indices=sentence_gru_out.sorted_indices,
                                                                 unsorted_indices=sentence_gru_out.unsorted_indices),
                                                  batch_first=True)  # (n_sentences, max(words_per_sentence))

            # Calculate softmax values as now words are arranged in their respective sentences
            sent_alphas = sent_att_out / torch.sum(sent_att_out, dim=1,
                                                   keepdim=True)  # (n_sentences, max(words_per_sentence))

            # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
            _paragraph, _ = pad_packed_sequence(sentence_gru_out,
                                                batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

            # Find sentence embeddings
            # gets the representation for the sentence
            _paragraph = (_paragraph.float() * sent_alphas.unsqueeze(2)).sum(
                dim=1)  # (batch_size, self.word_gru_out_size)

            paragraphs[:, paragraph_idx] = _paragraph
        # attention over paragraphs
        # paragraphs
        # pack padded sequence of sentences
        packed_paragraphs = pack_padded_sequence(paragraphs,
                                                 lengths=paragraphs_per_document_current_document.tolist(),
                                                 batch_first=True, enforce_sorted=False)
        paragraph_gru_out, _ = self.paragraph_gru(packed_paragraphs)
        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        paragraph_att_out = torch.tanh(self.paragraph_attention(paragraph_gru_out.data))
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        paragraph_att_out = self.paragraph_context_vector(paragraph_att_out).squeeze(1)  # (n_words)
        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence
        # First, take the exponent
        max_value = paragraph_att_out.max()  # scalar, for numerical stability during exponent calculation
        paragraph_att_out = torch.exp(paragraph_att_out - max_value)  # (n_words)
        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        paragraph_att_out, _ = pad_packed_sequence(PackedSequence(data=paragraph_att_out,
                                                                  batch_sizes=paragraph_gru_out.batch_sizes,
                                                                  sorted_indices=paragraph_gru_out.sorted_indices,
                                                                  unsorted_indices=paragraph_gru_out.unsorted_indices),
                                                   batch_first=True)  # (n_sentences, max(words_per_sentence))
        # Calculate softmax values as now words are arranged in their respective sentences
        paragraph_alphas = paragraph_att_out / torch.sum(paragraph_att_out, dim=1,
                                                         keepdim=True)  # (n_sentences, max(words_per_sentence))
        # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
        doc, _ = pad_packed_sequence(paragraph_gru_out,
                                     batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        # Find document embeddings
        doc = (doc.float() * paragraph_alphas.unsqueeze(2)).sum(dim=1)  # (batch_size, self.paragraph_gru_out_size)
        return doc

    def get_padded_document_structures(self, words_ids_a):
        document_structures = [self.training_dataset.get_document_structure(document) for document in words_ids_a]
        words_per_sentences_tensor = torch.LongTensor([
            self.training_dataset.get_padded_words_per_sentence(document_structure['words_per_sentence']) for
            document_structure in document_structures])
        sentences_per_paragraph_tensor = torch.LongTensor([
            self.training_dataset.get_padded_sentence_per_paragraph(document_structure['sentences_per_paragraph']) for
            document_structure in document_structures])

        paragraphs_per_document_tensor = torch.LongTensor(
            [document_structure['paragraphs_per_document'] for document_structure in document_structures])

        return words_per_sentences_tensor, sentences_per_paragraph_tensor, paragraphs_per_document_tensor

    def train(self):
        training_params = {'batch_size': self.batch_size,
                           'shuffle': True,
                           'drop_last': True}

        training_generator = DataLoader(self.dataset, **training_params)

        for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in training_generator:
            self.forward(current_document, words_per_sentence_current_document,
                         sentences_per_paragraph_current_document, paragraphs_per_document_current_document,
                         previous_document, words_per_sentence_previous_document,
                         sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document,
                         click_rate_tensor)


if __name__ == '__main__':
    model = SmashRNN()
    model.train()
