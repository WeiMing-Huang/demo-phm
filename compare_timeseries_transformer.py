'''https://github.com/KasperGroesLudvigsen/influenza_transformer/blob/main/transformer_timeseries.py'''

from torch import nn, Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from einops import rearrange, reduce, repeat


class PositionalEncoder(nn.Module):

    def __init__(
        self,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = True
    ):

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))

        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)

            pe[0, :, 0::2] = torch.sin(position * div_term)

            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)

            pe[:, 0, 0::2] = torch.sin(position * div_term)

            pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):

    def __init__(self,
                 input_size: int,
                 batch_first: bool,
                 dec_seq_len: int = 30,
                 dim_val: int = 512,
                 n_encoder_layers: int = 8,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.1,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 512,
                 num_predicted_features: int = 1
                 ):

        super().__init__()


        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            max_seq_len=dec_seq_len,
            dropout=dropout_pos_enc
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )


    def forward(self, src: Tensor) -> Tensor:


        src = self.encoder_input_layer(src)

        src = self.positional_encoding_layer(src)

        src = self.encoder(src=src)

        src = reduce(src, 'b n e -> b e', reduction='mean')

        output = self.linear_mapping(src)


        return output
