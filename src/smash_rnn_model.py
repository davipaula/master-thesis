"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

from sent_att_model import SentAttNet
from word_att_model import WordAttNet
from paragraph_att_model import ParagraphAttNet
from siamese_lstm import SiameseLSTM
import numpy as np


class SmashRNNModel(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, paragraph_hidden_size, batch_size, num_classes,
                 pretrained_word2vec_path, max_sent_length, max_word_length):
        super(SmashRNNModel, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length

        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self.paragraph_att_net = ParagraphAttNet(paragraph_hidden_size, sent_hidden_size, num_classes)
        self.siamese_lstm = SiameseLSTM()
        self._init_hidden_state()

        self.input_dim = 2 * paragraph_hidden_size * 3  # 3 = number of concatenations

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

        torch.set_printoptions(threshold=10000)

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size

        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.paragraph_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)

        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def siamese(self):
        print()

    def forward(self, current_document, current_document_structure, previous_document, previous_document_structure):

        # Adapt both functions to use document structure
        # word_representation_output = self.get_word_representation(current_document)
        #
        # sentence_representation_output = self.get_sentence_representation(current_document)

        # Generate representations at word, sentence and paragraph level. This is the MASH part of the model
        current_document_representation = self.get_document_representation(current_document, current_document_structure)
        previous_document_representation = self.get_document_representation(previous_document, previous_document_structure)

        # Concatenates document representations. This is the siamese part of the model
        concatenated_documents_representation = torch.cat((current_document_representation,
                                                           previous_document_representation,
                                                           torch.abs(
                                                               current_document_representation - previous_document_representation),
                                                           ), 1)

        predicted_ctr = self.classifier(concatenated_documents_representation)

        return predicted_ctr

    def get_document_representation(self, document, document_structure):
        paragraphs_per_document = document_structure['paragraphs_per_document']
        sentences_per_paragraph = document_structure['sentences_per_paragraph']
        words_per_sentence = document_structure['words_per_sentence']

        sentence_output_list = []

        for i, paragraph in enumerate(document):
            word_output_list = []

            # for j, sentence in enumerate(paragraph):
            #     if sentence.sum() == 0:
            #         break
            #
            #     words_in_sentence = words_per_sentence[i].squeeze(0)[j].unsqueeze(0)
            #
            #     word_output = self.word_att_net(sentence, words_in_sentence)
            #     word_output_list.append(word_output)

            # Re-arrange as sentences by removing sentence-pads (DOCUMENTS -> SENTENCES)
            # packed_sentences = pack_padded_sequence(documents,
            #                                         lengths=sentences_per_document.tolist(),
            #                                         batch_first=True,
            #                                         enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentences (n_sentences, word_pad_len)

            # Re-arrange as sentences by removing sentence-pads (DOCUMENTS -> SENTENCES)
            packed_sentences = pack_padded_sequence(document,
                                                    lengths=sentences_per_paragraph,
                                                    batch_first=True,
                                                    enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentences (n_sentences, word_pad_len)

            word_output = self.word_att_net(paragraph, words_per_sentence[i].squeeze(0))

            re_packed_sentence = PackedSequence(data=word_output,
                                                batch_sizes=packed_sentences.batch_sizes,
                                                sorted_indices=packed_sentences.sorted_indices,
                                                unsorted_indices=packed_sentences.unsorted_indices)  # a PackedSequence object, where 'data' is the output of the RNN (n_sentences, 2 * sentence_rnn_size)

            # word_output = torch.tensor(word_output_list)

            sentence_output = self.sent_att_net(re_packed_sentence)
            sentence_output_list.append(sentence_output)

        sentence_output = torch.cat(sentence_output_list, 0)
        output, _ = self.paragraph_att_net(sentence_output, None)
        return output

    def get_word_representation(self, document):
        # Removes paragraphs and sentences structures, transforming the document into a long sequence of words
        document = document.view(1, np.prod(document.shape))

        word_output = self.word_att_net(document)

        return word_output

    def get_sentence_representation(self, document):
        # Removes the sentences structure, transforming the document into a set of paragraphs containing
        # a long sequence of words
        document = document.view(document.shape[0], document.shape[1], document.shape[2] * document.shape[3])

        sentence_output = self.get_document_representation(document)

        return sentence_output
