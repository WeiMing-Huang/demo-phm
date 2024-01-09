'''https://github.com/Renovamen/Text-Classification'''
import torch
from torch import nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, bidirectional):
        super(BiLSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True,
                            bidirectional=bidirectional)
        if self.bidirectional:
            self.classifier = nn.Linear(hidden_dim * 2, n_class)
        else:
            self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):

        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            out = torch.hstack((hn[-2, :, :], hn[-1, :, :]))
        else:
            out = out[:, -1, :]
        out = self.classifier(out)
        return out
