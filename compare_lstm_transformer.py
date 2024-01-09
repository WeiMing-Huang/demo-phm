#%%

import torch
import torch.nn as nn

class LstmTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, out_size):
        super(LstmTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=hidden_size*4, dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = self.transformer(x, x)
        x = x.transpose(0, 1)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

#%%
