"""Calibrate the discriminator's density-ratio estimate on held-out data (Platt scaling).

A discriminator that separates real and fake too well gives r(x) = D/(1-D)
squashed towards 0 (E_{p_hat}[r] << 1): it then measures 'does it look real?'
rather than p/p_hat, and OBRS over-selects easy modes. We refit its logit s(x):

    D_cal(x) = sigmoid(a * s(x) + b)   =>   r(x) = exp(a * s(x) + b)

by logistic regression on HELD-OUT data: the MNIST test set (never used to
train D) vs fresh samples from the frozen G. Only a (the 'temperature') matters
for OBRS, since c_K absorbs any constant factor; b is kept for the diagnostic
E_{p_hat}[r] ~ 1.  Saves checkpoints/<D_name without .pth>_calib.json.
Usage:  python calibrate_D.py --D_name D_ft.pth
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, transforms

from model import Generator, Discriminator
from utils import load_model
from dot import d_logit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--G_name', default='G.pth')
    parser.add_argument('--D_name', default='D_ft.pth')
    parser.add_argument('--n', type=int, default=10000, help='samples per class')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = load_model(Generator(784).to(device), 'checkpoints', device, name=args.G_name).eval()
    D = load_model(Discriminator(784).to(device), 'checkpoints', device, name=args.D_name).eval()

    data_path = os.getenv('DATA') or 'data'
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test = datasets.MNIST(data_path, train=False, download=False, transform=tf)
    real = torch.stack([test[i][0] for i in range(min(args.n, len(test)))]).view(-1, 784)

    with torch.no_grad():
        s_real = torch.cat([d_logit(D, real[i:i + 2000].to(device)).cpu()
                            for i in range(0, len(real), 2000)])
        s_fake = torch.cat([d_logit(D, G(torch.randn(2000, 100, device=device))).cpu()
                            for _ in range(len(real) // 2000)])

    s = torch.cat([s_real, s_fake]).numpy().reshape(-1, 1)
    y = np.concatenate([np.ones(len(s_real)), np.zeros(len(s_fake))])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s, y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])

    raw_r = torch.exp(s_fake).mean().item()
    cal_r = torch.exp(a * s_fake + b).mean().item()
    acc = ((torch.cat([s_real, s_fake]) > 0).float().numpy() == y).mean()
    print(f'held-out accuracy of D: {acc:.3f}')
    print(f'calibration: a (temperature) = {a:.4f}, b = {b:.4f}')
    print(f'E_fake[r] before = {raw_r:.3f}, after = {cal_r:.3f}  (target ~1)')

    out = os.path.join('checkpoints', args.D_name.replace('.pth', '') + '_calib.json')
    json.dump({'a': a, 'b': b, 'D_name': args.D_name, 'G_name': args.G_name,
               'heldout_acc': float(acc), 'mean_r_raw': raw_r, 'mean_r_cal': cal_r},
              open(out, 'w'), indent=1)
    print(f'Saved {out}')
