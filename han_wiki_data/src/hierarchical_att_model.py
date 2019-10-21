"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from sent_att_model import SentAttNet
from word_att_model import WordAttNet
from paragraph_att_model import ParagraphAttNet


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, paragraph_hidden_size, batch_size, num_classes,
                 pretrained_word2vec_path, max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self.paragraph_att_net = ParagraphAttNet(paragraph_hidden_size, sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()

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

    def forward(self, input):

        word_output_list, sentence_output_list = [], []

        input = input.permute(1, 0, 2, 3)

        for paragraph in input:
            for word in paragraph:
                word_output, self.word_hidden_state = self.word_att_net(word.permute(1, 0), self.word_hidden_state)
                word_output_list.append(word_output)

            word_output = torch.cat(word_output_list, 0)

            sentence_output, self.sent_hidden_state = self.sent_att_net(word_output.permute(1, 0, 2),
                                                                        self.sent_hidden_state)
            sentence_output_list.append(sentence_output)

        # for paragraph in input:

        sentence_output = torch.cat(sentence_output_list, 1)

        output, self.paragraph_hidden_state = self.paragraph_att_net(sentence_output, self.paragraph_hidden_state)

        return output
