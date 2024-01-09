import torch
from torch import nn    

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim,  n_layers, output_dim=1, drop_prob=0.02):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(p=0.3)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        return hidden

    
    def forward(self, x):

        h = self.init_hidden(x.size(0))

        out, h = self.gru(x, h)

        out = self.drop(out)

        out = torch.squeeze(out)

        out = out.mean(dim=1)
        
        out = self.linear(out)
        
        return out