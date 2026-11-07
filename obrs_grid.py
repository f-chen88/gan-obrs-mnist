"""Figure for the slides: which samples does OBRS reject?

Draws n samples from G, computes the OBRS acceptance probability
a(x) = min(c_K r(x), 1) for budget K, and saves two grids:
  figures/obrs_rejected.png  - the 64 samples with the LOWEST a(x) (almost always rejected)
  figures/obrs_accepted.png  - the 64 samples with the HIGHEST a(x) (kept)
Usage:  python obrs_grid.py --K 2
"""
import argparse
import os

import torch
import torchvision

from model import Generator, Discriminator
from d_models import build_D
from utils import load_model
from obrs import calibrate, density_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=float, default=2.0)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--G_name', default='G.pth')
    parser.add_argument('--D_name', default='D_ft.pth')
    parser.add_argument('--D_arch', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--prefix', default='obrs', help='output file prefix in figures/')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = load_model(Generator(784).to(device), 'checkpoints', device, name=args.G_name).eval()
    D = load_model(build_D(args.D_arch).to(device), 'checkpoints', device, name=args.D_name).eval()

    c, _ = calibrate(G, D, args.K, device)
    with torch.no_grad():
        x = G(torch.randn(args.n, 100, device=device))
        a = torch.clamp(c * density_ratio(D, x), max=1.0)
    order = torch.argsort(a)
    rejected = x[order[:64]]
    kept = x[order[-64:]]                                   # highest a(x): kept (a = 1 if any)
    print(f'K={args.K}: c_K={c:.4g}; {(a >= 1).float().mean():.1%} of samples have a(x)=1, '
          f'lowest a(x) shown: {a[order[:64]].max():.3g}')

    os.makedirs('figures', exist_ok=True)
    for name, imgs in [(f'{args.prefix}_rejected', rejected), (f'{args.prefix}_accepted', kept)]:
        imgs = (imgs.view(-1, 1, 28, 28) + 1) / 2          # [-1,1] -> [0,1] for display
        torchvision.utils.save_image(imgs, f'figures/{name}.png', nrow=8, padding=2, pad_value=1)
    print(f'Saved figures/{args.prefix}_rejected.png and figures/{args.prefix}_accepted.png')
