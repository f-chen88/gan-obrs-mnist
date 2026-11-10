"""Discriminator architectures used as density-ratio estimators for OBRS.

'mlp' is the starter Discriminator from model.py (unchanged).
'cnn' is a small convolutional classifier with the same interface:
input (N, 784) in [-1, 1], output (N, 1) = probability that x is real.
Only the GENERATOR architecture is fixed by the assignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Discriminator


class DiscriminatorCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)    # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)   # 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.drop = nn.Dropout(dropout)

    def logit(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.drop(x.flatten(1))
        x = self.drop(F.leaky_relu(self.fc1(x), 0.2))
        return self.fc2(x)

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


SN = nn.utils.spectral_norm


class DiscriminatorMLP_SN(Discriminator):
    """The starter MLP discriminator with spectral normalisation on every layer."""
    def __init__(self):
        super().__init__(784)
        for name in ('fc1', 'fc2', 'fc3', 'fc4'):
            setattr(self, name, SN(getattr(self, name)))

    def logit(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return self.fc4(x)


class DiscriminatorCNN_SN(nn.Module):
    """DCGAN-style convolutional discriminator with spectral normalisation (no dropout, no BN)."""
    def __init__(self, width=64):
        super().__init__()
        self.conv1 = SN(nn.Conv2d(1, width, 4, stride=2, padding=1))              # 28 -> 14
        self.conv2 = SN(nn.Conv2d(width, 2 * width, 4, stride=2, padding=1))      # 14 -> 7
        self.conv3 = SN(nn.Conv2d(2 * width, 4 * width, 3, stride=1, padding=1))  # 7 -> 7
        self.fc = SN(nn.Linear(4 * width * 7 * 7, 1))

    def logit(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return self.fc(x.flatten(1))

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


ARCHS = ['mlp', 'mlp_sn', 'cnn', 'cnn_sn']


def build_D(arch='mlp'):
    if arch == 'mlp':
        return Discriminator(784)
    if arch == 'mlp_sn':
        return DiscriminatorMLP_SN()
    if arch == 'cnn':
        return DiscriminatorCNN()
    if arch == 'cnn_sn':
        return DiscriminatorCNN_SN()
    raise ValueError(f'unknown discriminator architecture: {arch}')


def d_logit(D, x):
    """Logit of D (before the sigmoid), shape (N,), for every architecture above."""
    if hasattr(D, 'logit'):
        return D.logit(x).squeeze(1)
    x = F.leaky_relu(D.fc1(x), 0.2)        # starter MLP
    x = F.leaky_relu(D.fc2(x), 0.2)
    x = F.leaky_relu(D.fc3(x), 0.2)
    return D.fc4(x).squeeze(1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
