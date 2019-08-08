from torch import nn


class SMASH(nn.Module):
    """

    Just a dummy

    """
    def __init__(self):
        super().__init__()

    def forward(self, features_a, features_b):

        # A
        # word attention
        # sent attention
        # (paragraph attention - leave out for now)

        # B
        # word attention
        # sent attention
        # (paragraph attention - leave out for now)

        # siamese: concat A - B

        # output (binary: similar or not)
        pass