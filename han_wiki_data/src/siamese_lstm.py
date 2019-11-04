from torch import nn
import torch
from lstm_encoder import LSTMEncoder


class SiameseLSTM(nn.Module):
    def __init__(self):
        super(SiameseLSTM, self).__init__()

        self.encoder = LSTMEncoder()
        self.fc_dim = 100

        # self.input_dim = 5 * self.encoder.direction * self.encoder.hidden_size
        self.input_dim = 5 * 1 * 150

        self.mlp_dim = int(self.input_dim / 2)
        self.out_dim = 2

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.Linear(self.mlp_dim, self.out_dim)
        )

    def forward(self, current_document, previous_document):
        # init hidden, cell
        hidden_current_document, cell_current_document = self.encoder.init_hidden_cell()
        hidden_previous_document, cell_previous_document = self.encoder.init_hidden_cell()

        # input one by one
        v1, hidden_current_document, cell_current_document = self.encoder(current_document.unsqueeze(0),
                                                                          hidden_current_document,
                                                                          cell_current_document)

        v2, hidden_previous_document, cell_previous_document = self.encoder(previous_document.unsqueeze(0),
                                                                            hidden_previous_document,
                                                                            cell_previous_document)

        # utilize these two encoded vectors
        features = torch.cat((v1, torch.abs(v1 - v2), v2, v1 * v2, (v1 + v2) / 2), 2)
        # features = v1-v2
        output = self.classifier(features)

        return output


if __name__ == '__main__':
    abc = SiameseLSTM()
