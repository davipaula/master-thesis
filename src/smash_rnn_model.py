"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
import numpy as np
import pandas as pd
import csv

from src.utils import remove_zero_tensors_from_batch, remove_zeros_from_words_per_sentence, \
    add_filtered_tensors_to_original_batch


class SmashRNNModel(nn.Module):
    def __init__(self, dict, dict_len, embedding_size, max_word_length, max_sent_length, max_paragraph_length):
        super(SmashRNNModel, self).__init__()

        self.max_word_length = max_word_length
        self.max_sent_length = max_sent_length
        self.max_paragraph_length = max_paragraph_length
        self.batch_size = 1

        # Init embedding layer
        self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embedding_size).from_pretrained(dict)

        # RNN + attention layers
        word_gru_hidden_size = 100
        self.word_gru_out_size = word_gru_hidden_size * 2
        self.word_gru = nn.GRU(embedding_size, word_gru_hidden_size, bidirectional=True, batch_first=True)
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

    def get_document_representation(self, document, paragraphs_per_document, sentences_per_paragraph,
                                    words_per_sentence):

        batch_size = document.shape[0]
        max_paragraphs_per_document = max(paragraphs_per_document.sum(dim=1))
        max_sentences_per_paragraph = max(sentences_per_paragraph.sum(dim=1))

        # zero placeholders
        sentences = torch.zeros((batch_size, max_sentences_per_paragraph, self.word_gru_out_size))
        paragraphs = torch.zeros((batch_size, max_paragraphs_per_document, self.sentence_gru_out_size))

        if torch.cuda.is_available():
            sentences = sentences.cuda()
            paragraphs = paragraphs.cuda()

        # iterate over each hierarchy level
        for paragraph_idx in range(max_paragraphs_per_document):
            for sentence_idx in range(max(sentences_per_paragraph[:, paragraph_idx])):
                sentences_in_batch = document[:, paragraph_idx, sentence_idx, :]

                word_ids_in_sentences = remove_zero_tensors_from_batch(sentences_in_batch)

                words_per_sentence_in_paragraph = remove_zeros_from_words_per_sentence(
                    words_per_sentence[:, paragraph_idx, sentence_idx])

                # get word embeddings from ids
                word_embeddings = self.embedding(word_ids_in_sentences)

                packed_word_embeddings = pack_padded_sequence(word_embeddings,
                                                              lengths=words_per_sentence_in_paragraph,
                                                              batch_first=True,
                                                              enforce_sorted=False).float()

                try:
                    word_level_gru, _ = self.word_gru(packed_word_embeddings)
                except:
                    print('error')

                word_level_attention = self.get_word_attention(word_level_gru)

                # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
                # (n_sentences, max(words_per_sentence))
                word_level_attention, _ = pad_packed_sequence(PackedSequence(data=word_level_attention,
                                                                             batch_sizes=word_level_gru.batch_sizes,
                                                                             sorted_indices=word_level_gru.sorted_indices,
                                                                             unsorted_indices=word_level_gru.unsorted_indices),
                                                              batch_first=True)

                word_level_attention = add_filtered_tensors_to_original_batch(word_level_attention, sentences_in_batch)

                word_level_alphas = self.get_alphas(word_level_attention)

                # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
                word_level_gru, _ = pad_packed_sequence(word_level_gru, batch_first=True)

                try:
                    word_level_gru = add_filtered_tensors_to_original_batch(word_level_gru, sentences_in_batch)
                except:
                    print('problem')

                try:
                    sentences[:, sentence_idx] = self.get_representation(word_level_alphas, word_level_gru)
                except:
                    print('problem')

            # pack padded sequence of sentences
            packed_sentences = pack_padded_sequence(sentences,
                                                    # TODO refactor this monstruosity
                                                    lengths=sentences_per_paragraph.sum(dim=1).tolist(),
                                                    batch_first=True, enforce_sorted=False)

            sentence_level_gru, _ = self.sentence_gru(packed_sentences)

            sentence_level_attention = self.get_sentence_level_attention(sentence_level_gru)

            # Calculate softmax values as now words are arranged in their respective sentences
            sentence_alphas = self.get_alphas(sentence_level_attention)

            # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
            sentence_level_gru, _ = pad_packed_sequence(sentence_level_gru, batch_first=True)

            paragraphs[:, paragraph_idx] = self.get_representation(sentence_alphas, sentence_level_gru)

        # attention over paragraphs
        # paragraphs
        # pack padded sequence of sentences
        packed_paragraphs = pack_padded_sequence(paragraphs,
                                                 lengths=paragraphs_per_document.squeeze(1),
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

    def get_sentence_level_attention(self, sentence_level_gru):
        sentence_level_attention = torch.tanh(self.sentence_attention(sentence_level_gru.data))
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        sentence_level_attention = self.sentence_context_vector(sentence_level_attention).squeeze(1)  # (n_words)
        sentence_level_attention = self.softmax(sentence_level_attention)
        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentence_level_attention, _ = pad_packed_sequence(PackedSequence(data=sentence_level_attention,
                                                                         batch_sizes=sentence_level_gru.batch_sizes,
                                                                         sorted_indices=sentence_level_gru.sorted_indices,
                                                                         unsorted_indices=sentence_level_gru.unsorted_indices),
                                                          batch_first=True)  # (n_sentences, max(words_per_sentence))
        return sentence_level_attention

    def softmax(self, sentence_level_attention):
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
        packed_word_embeddings = pack_padded_sequence(word_embeddings,
                                                      lengths=words_per_sentence_in_paragraph,
                                                      batch_first=True,
                                                      enforce_sorted=False).float()

        word_level_gru, _ = self.word_gru(packed_word_embeddings)

        return word_level_gru

    @staticmethod
    def get_alphas(attention_representation):
        alphas = attention_representation / torch.sum(attention_representation, dim=1, keepdim=True)
        alphas = torch.where(torch.isnan(alphas), torch.zeros_like(alphas), alphas)

        return alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence))


if __name__ == '__main__':
    word2vec_path = '../data/glove.6B.50d.txt'
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
    SmashRNNModel(dict, dict_len, embed_dim)
