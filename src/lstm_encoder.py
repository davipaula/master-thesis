from torch import nn
from torch.autograd import Variable
import torch


class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()

        self.embed_size = 26
        self.batch_size = 24
        self.hidden_size = 150
        self.num_layers = 24
        self.bidir = False
        self.direction = 2 if self.bidir else 1
        self.dropout = 0
        # self.embedding = config['embedding_matrix']

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, dropout=self.dropout,
                            num_layers=self.num_layers, bidirectional=self.bidir)

    def init_hidden_cell(self):
        rand_hidden = Variable(torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size))

        return rand_hidden, rand_cell

    def forward(self, input, hidden, cell):
        # input = self.embedding(input).view(1, 1, -1)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))

        return output, hidden, cell


if __name__ == '__main__':
    abc = LSTMEncoder()
