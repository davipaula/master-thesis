"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from utils import matrix_mul, element_wise_mul


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        self.sentence_attention = nn.Linear(2 * word_hidden_size, sent_hidden_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(sent_hidden_size, 1,
                                                 bias=False)  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector

        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input):
        sentence_representation, _ = self.gru(input, None)

        att_s = self.sentence_attention(sentence_representation.data)  # (n_sentences, att_size)
        att_s = torch.tanh(att_s)  # (n_sentences, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s).squeeze(1)  # (n_sentences)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over sentences in the same document

        # First, take the exponent
        max_value = att_s.max()  # scalar, for numerical stability during exponent calculation
        att_s = torch.exp(att_s - max_value)  # (n_sentences)

        # Re-arrange as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        att_s, _ = pad_packed_sequence(PackedSequence(data=att_s,
                                                      batch_sizes=sentence_representation.batch_sizes,
                                                      sorted_indices=sentence_representation.sorted_indices,
                                                      unsorted_indices=sentence_representation.unsorted_indices),
                                       batch_first=True)  # (n_documents, max(sentences_per_document))

        # Calculate softmax values as now sentences are arranged in their respective documents
        sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)  # (n_documents, max(sentences_per_document))

        # Similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        paragraph, _ = pad_packed_sequence(sentence_representation,
                                           batch_first=True)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)

        return paragraph


if __name__ == "__main__":
    abc = SentAttNet()


























