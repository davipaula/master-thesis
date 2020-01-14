"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
import numpy as np
import pandas as pd
import csv
from src.utils import get_document_at_sentence_level, get_words_per_sentence_at_sentence_level


class SentenceSmashRNNModel(nn.Module):
    def __init__(self, dict, dict_len, embedding_size, max_word_length, max_sent_length):
        super(SentenceSmashRNNModel, self).__init__()

        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        self.max_word_length = max_word_length
        self.max_sent_length = max_sent_length
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

        self.input_dim = 2 * sentence_gru_hidden_size * 3  # 3 = number of concatenations

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

    def forward(self, current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document,
                previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document,
                click_rate_tensor):

        current_document = get_document_at_sentence_level(current_document)
        previous_document = get_document_at_sentence_level(previous_document)

        words_per_sentence_current_document = get_words_per_sentence_at_sentence_level(
            words_per_sentence_current_document)
        words_per_sentence_previous_document = get_words_per_sentence_at_sentence_level(
            words_per_sentence_previous_document)

        # This only works with self.batch_size = 1
        current_document_representation = self.get_document_representation(current_document,
                                                                           sentences_per_paragraph_current_document,
                                                                           words_per_sentence_current_document)
        previous_document_representation = self.get_document_representation(previous_document,
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

    def get_document_representation(self, document, sentences_per_paragraph, words_per_sentence):

        # this only works with batch_size = 1
        _sentences_per_paragraph = np.sum([[words for words in paragraph if words > 0] for paragraph in
                                    sentences_per_paragraph.tolist()][0])
        # zero placeholders
        sentences = torch.zeros((self.batch_size, _sentences_per_paragraph, self.word_gru_out_size))

        if torch.cuda.is_available():
            sentences = sentences.cuda()

        # docs = torch.zeros((self.batch_size, self.paragraph_gru_out_size))
        # iterate over each hierarchy level
        for sentence_idx in range(_sentences_per_paragraph):
            # attention over words
            word_ids_in_sent = document[:, sentence_idx, :]
            # 1st dim = batch, last dim = words
            words_in_sent = self.embedding(word_ids_in_sent)  # get word embeddings from ids

            # pack padded sequence
            packed_words = pack_padded_sequence(words_in_sent,
                                                lengths=[words_per_sentence[sentence_idx].tolist()],
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
            _sentence = (_sentence.float() * word_alphas.unsqueeze(2)).sum(dim=1)  # (batch_size, self.word_gru_out_size)

            sentences[:, sentence_idx] = _sentence

        # attention over sentences
        sentences_in_paragraph = sentences

        # pack padded sequence of sentences
        packed_sentences = pack_padded_sequence(sentences_in_paragraph,
                                                # TODO refactor this monstruosity
                                                lengths=[np.sum(sentences_per_paragraph.tolist())],
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
        paragraph_representation, _ = pad_packed_sequence(sentence_gru_out, batch_first=True)

        # gets the representation for the paragraph
        paragraph_representation = (paragraph_representation * sent_alphas.unsqueeze(2)).sum(
            dim=1)  # (batch_size, self.word_gru_out_size)

        return paragraph_representation

    def test_model(self):
        training_generator = torch.load('../data/training.pth')

        for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in training_generator:
            if torch.cuda.is_available():
                current_document = current_document.cuda()
                words_per_sentence_current_document = words_per_sentence_current_document.cuda()
                sentences_per_paragraph_current_document = sentences_per_paragraph_current_document.cuda()
                paragraphs_per_document_current_document = paragraphs_per_document_current_document.cuda()
                previous_document = previous_document.cuda()
                words_per_sentence_previous_document = words_per_sentence_previous_document.cuda()
                sentences_per_paragraph_previous_document = sentences_per_paragraph_previous_document.cuda()
                paragraphs_per_document_previous_document = paragraphs_per_document_previous_document.cuda()
                click_rate_tensor = click_rate_tensor.cuda()

            predictions = self.forward(current_document, words_per_sentence_current_document,
                                       sentences_per_paragraph_current_document,
                                       previous_document, words_per_sentence_previous_document,
                                       sentences_per_paragraph_previous_document,
                                       click_rate_tensor)

            print(predictions)

            break


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    word2vec_path = '../data/glove.6B.50d.txt'
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
    model = SentenceSmashRNNModel(dict, dict_len, embed_dim, 90, 50)
    model.test_model()
