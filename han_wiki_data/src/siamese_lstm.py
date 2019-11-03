from torch import nn
import torch
import lstm_encoder as LSTMEncoder


class SiameseLSTM(nn.Module):
    def __init__(self):
        super(SiameseLSTM, self).__init__()

        self.encoder = LSTMEncoder
        self.fc_dim = 100

        # self.input_dim = 5 * self.encoder.direction * self.encoder.hidden_size
        self.input_dim = 5 * 1 * 150

        self.mlp_dim = int(self.input_dim / 2)
        self.out_dim = 2

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.input_dim, self.fc_dim),
        #     nn.Linear(self.fc_dim, 2)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.Linear(self.mlp_dim, self.out_dim)
        )

    def forward(self, s1, s2):
        # init hidden, cell
        h1, c1 = self.encoder.initHiddenCell()
        h2, c2 = self.encoder.initHiddenCell()

        # input one by one

        for i in range(len(s1)):
            v1, h1, c1 = self.encoder(s1[i], h1, c1)

        for j in range(len(s2)):
            v2, h2, c2 = self.encoder(s2[j], h2, c2)

        # utilize these two encoded vectors
        features = torch.cat((v1, torch.abs(v1 - v2), v2, v1*v2, (v1+v2)/2), 2)
        # features = v1-v2
        output = self.classifier(features)

        return output


if __name__ == '__main__':
    abc = SiameseLSTM()
