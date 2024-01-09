import torch
from torch import nn    

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.drop = nn.Dropout(p=0.4)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    
    def init_hidden(self, x):
        h0 = torch.zeros((self.layer_dim, x.size(0), self.hidden_dim)).cuda()
        c0 = torch.zeros((self.layer_dim, x.size(0), self.hidden_dim)).cuda()
        return h0, c0
    
    
    def forward(self, x):

        self.hn, self.cn = self.init_hidden(x)

        out, (self.hn, self.cn) = self.lstm(x, (self.hn[:, :x.size(0), :].detach(), self.cn[:, :x.size(0), :].detach()))

        out = self.drop(out) 

        out = torch.squeeze(out)

        out = out.mean(dim=1)

        out = self.linear(out)

        return out