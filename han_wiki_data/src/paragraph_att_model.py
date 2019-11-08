"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matrix_mul, element_wise_mul


class ParagraphAttNet(nn.Module):
    def __init__(self, paragraph_hidden_size=14, sent_hidden_size=62, num_classes=14):
        super(ParagraphAttNet, self).__init__()
        # TODO the parameters are hardcoded - paragraph_hidden_size and sent_hidden_size match both

        self.paragraph_weight = nn.Parameter(torch.Tensor(2 * paragraph_hidden_size, 2 * paragraph_hidden_size))
        self.paragraph_bias = nn.Parameter(torch.Tensor(1, 2 * paragraph_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * paragraph_hidden_size, 1))

        self.gru = nn.GRU(sent_hidden_size, paragraph_hidden_size, bidirectional=True)
        self.fc = nn.Linear(paragraph_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.paragraph_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.paragraph_weight, self.paragraph_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)

        # TODO should this model have an FC? If so, how to configure it?
        # output = self.fc(output)

        return output, h_output


if __name__ == "__main__":
    abc = ParagraphAttNet()
