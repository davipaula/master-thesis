"""
@author: Davi Nascimento de Paula <davi.paula@gmail.com>
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from src.utils import get_document_at_word_level


class WordLevelSmashRNNModel(nn.Module):
    def __init__(self, dict, dict_len, embedding_size):
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
        current_document, words_in_current_document = get_document_at_word_level(current_document,
                                                                                 words_per_sentence_current_document)
        previous_document, words_in_previous_document = get_document_at_word_level(previous_document,
                                                                                   words_per_sentence_previous_document)

        current_document_representation = self.get_document_representation(current_document, words_in_current_document)
        previous_document_representation = self.get_document_representation(previous_document,
                                                                            words_in_previous_document)

        # Concatenates document representations. This is the siamese part of the model
        concatenated_documents_representation = torch.cat((current_document_representation,
                                                           previous_document_representation,
                                                           torch.abs(
                                                               current_document_representation - previous_document_representation),
                                                           ), 1)

        predicted_ctr = self.classifier(concatenated_documents_representation)

        return predicted_ctr

    def get_document_representation(self, word_ids_in_sentence, words_in_sentence):
        if torch.cuda.is_available():
            word_ids_in_sentence = word_ids_in_sentence.cuda()

        # attention over words
        # 1st dim = batch, last dim = words
        # get word embeddings from ids
        word_embeddings = self.embedding(word_ids_in_sentence)

        packed_word_embeddings = pack_padded_sequence(word_embeddings,
                                                      lengths=words_in_sentence,
                                                      batch_first=True,
                                                      enforce_sorted=False)

        word_gru_output, _ = self.word_gru(packed_word_embeddings.float())

        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        word_attention_output = torch.tanh(self.word_attention(word_gru_output.data))

        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        word_attention_output = self.word_context_vector(word_attention_output).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = word_attention_output.max()  # scalar, for numerical stability during exponent calculation
        word_attention_output = torch.exp(word_attention_output - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        word_attention_output, _ = pad_packed_sequence(PackedSequence(data=word_attention_output,
                                                                      batch_sizes=word_gru_output.batch_sizes,
                                                                      sorted_indices=word_gru_output.sorted_indices,
                                                                      unsorted_indices=word_gru_output.unsorted_indices),
                                                       batch_first=True)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = word_attention_output / torch.sum(word_attention_output, dim=1,
                                                        keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentence by re-padding with 0s (WORDS -> SENTENCES)
        _sentence, _ = pad_packed_sequence(word_gru_output,
                                           batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        # gets the representation for the sentence
        sentence_representation = (_sentence.float() * word_alphas.unsqueeze(2)).sum(dim=1)
        # (batch_size, self.word_gru_out_size)

        return sentence_representation

    def test_forward(self):
        print('Test a forward step')

        training_generator = torch.load('../data/training.pth')

        for current_document, words_per_sentence_current_document, sentences_per_paragraph_current_document, paragraphs_per_document_current_document, previous_document, words_per_sentence_previous_document, sentences_per_paragraph_previous_document, paragraphs_per_document_previous_document, click_rate_tensor in training_generator:
            prediction = self.forward(current_document,
                                      words_per_sentence_current_document,
                                      previous_document,
                                      words_per_sentence_previous_document,
                                      click_rate_tensor)

            print(prediction)
            print(click_rate_tensor)

            break


if __name__ == '__main__':
    word2vec_path = '../preparation/glove.6B.50d.txt'
    dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
    dict_len, embed_dim = dict.shape
    dict_len += 1
    unknown_word = np.zeros((1, embed_dim))
    dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
    word_level_model = WordLevelSmashRNNModel(dict, dict_len, embed_dim)
    word_level_model.test_forward()
