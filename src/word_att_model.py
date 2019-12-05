"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv


class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.word_hidden_state = torch.zeros(2, 1, hidden_size)

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self._create_weights(mean=0.0, std=0.05)

        self.word_attention = nn.Linear(2 * hidden_size, 50)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(50, 1, bias=False)

        torch.set_printoptions(threshold=10000)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def convert_tensor_to_list(self, tensor):
        converted_list = []

        for tensor_item in tensor[0]:
            converted_list.append(tensor_item.numpy()[0])

        return converted_list

    def forward(self, sentence, words_per_sentence):
        if len(sentence.shape) == 1:
            sentence = sentence.view(1, 54)

        sentence_embeddings = self.lookup(sentence)

        packed_sentence_embeddings = pack_padded_sequence(sentence_embeddings,
                                                          lengths=words_per_sentence.tolist(),
                                                          batch_first=True,
                                                          enforce_sorted=False)

        words_representation, _ = self.gru(packed_sentence_embeddings.float())
        # This implementation uses the feature sentence_embeddings. Paper uses hidden state
        word_attention = self.word_attention(words_representation.data)
        word_attention = torch.tanh(word_attention)

        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        word_attention = self.word_context_vector(word_attention).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = word_attention.max()  # scalar, for numerical stability during exponent calculation
        word_attention = torch.exp(word_attention - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        word_attention, _ = pad_packed_sequence(PackedSequence(data=word_attention,
                                                               batch_sizes=words_representation.batch_sizes,
                                                               sorted_indices=words_representation.sorted_indices,
                                                               unsorted_indices=words_representation.unsorted_indices),
                                                batch_first=True)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = word_attention / torch.sum(word_attention, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(words_representation,
                                           batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences.float() * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # gets the representation for the sentence
        sentences = sentences.sum(dim=1)  # (n_sentences)

        return sentences


def test_model():
    sentence = torch.LongTensor([8, 9354, 31, 8, 301, 39, 13856, 154, 47, 65, 4, 27, 215, 5, 8, 2831, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0])

    sentence = sentence.view(1, 54)

    word_model = WordAttNet("../data/glove.6B.50d.txt")

    output = word_model(sentence, [16])

    print(output)


if __name__ == "__main__":
    test_model()
