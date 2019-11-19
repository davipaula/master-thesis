"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
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

        self.input_dim = 2 * paragraph_hidden_size * 3   # 3 = number of concatenations

        # Not mentioned in the paper.
        self.mlp_dim = int(self.input_dim / 2)
        self.out_dim = 1

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.out_dim)
        )

        self.sigmoid = nn.Sigmoid()

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

    def forward(self, current_document, previous_document):

        # word_representation_output = self.get_word_representation(current_document)

        sentence_representation_output = self.get_sentence_representation(current_document)

        # Generate representations at word, sentence and paragraph level
        current_document_output = self.get_document_representation(current_document)
        previous_document_output = self.get_document_representation(previous_document)

        # utilize these two encoded vectors
        features = torch.cat((current_document_output,
                              previous_document_output,
                              torch.abs(current_document_output - previous_document_output),
                              ), 1)

        output = self.classifier(features)

        predicted_ctr = self.sigmoid(output)

        return predicted_ctr

    def get_document_representation(self, document):
        sentence_output_list = []

        for paragraph in document:
            word_output_list = []

            for word in paragraph:
                word_output, self.word_hidden_state = self.word_att_net(word, self.word_hidden_state)
                word_output_list.append(word_output)

            word_output = torch.cat(word_output_list, 0)

            sentence_output, self.sent_hidden_state = self.sent_att_net(word_output,
                                                                        self.sent_hidden_state)
            sentence_output_list.append(sentence_output)
        # for paragraph in input:
        sentence_output = torch.cat(sentence_output_list, 0)
        output, self.paragraph_hidden_state = self.paragraph_att_net(sentence_output,
                                                                     self.paragraph_hidden_state)
        return output

    def get_word_representation(self, document):
        # Transforms document into a long sequence of words
        document = document.view(1, np.prod(document.shape))

        word_output, _ = self.word_att_net(document, self.word_hidden_state)

        return word_output

    def get_sentence_representation(self, document):
        # Transforms paragraphs into a sequence of words
        document = document.view(document.shape[0], document.shape[1], document.shape[2] * document.shape[3])

        sentence_output = self.get_document_representation(document)

        return sentence_output
