# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from compare_vision_transformer import ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64, out_dim=128):  # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):

    def __init__(self, backbone=ViT(), backbone_out=256):

        super().__init__()

        self.backbone = backbone

        hidden = backbone_out

        self.projector = projection_MLP(hidden, hidden, hidden)

        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )

        self.predictor = prediction_MLP(hidden, int(hidden/2), hidden)

        self.tune = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.ssl = nn.Sequential(
            self.encoder,
            self.predictor,
        )

        self.final = nn.Sequential(
            self.encoder,
            self.predictor,
            self.tune
        )

    def forward(self, x1, x2, x3):

        f, h = self.encoder, self.predictor
        z1, z2, z3 = f(x1), f(x2), f(x3)
        p1, p2, p3 = h(z1), h(z2), h(z3)
        return z1, z2, z3, p1, p2, p3


if __name__ == "__main__":
    model = SimSiam().to(device)

    x1 = torch.randn((10, 1, 48, 10)).to(device)
    x2 = torch.randn_like(x1).to(device)

    z1, z2, p1, p2 = model(x1, x2)
    L = D(p1, z2) / 2 + D(p2, z1) / 2
    L.backward()

    print("forward backwork check")


# %%
