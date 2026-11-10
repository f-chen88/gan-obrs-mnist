"""Visual check of OBRS samples as the budget K grows (figure for the report/slides).

One fixed pool of latents and one fixed uniform draw u per sample, so the accepted
sets are nested: a sample kept at budget K is also kept at every smaller K.
Rows: K = 1, 2, 5, 10, 20, 50, then the K -> infinity limit (top samples by r).
Also plots the digit proportions of the accepted samples (eval_classifier.pth).
    python samples_vs_K.py --D_name D_ft1.pth --bg_thresh 0.2
Outputs: figures/<prefix>_grid.png, figures/<prefix>_classes.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision

from model import Generator
from d_models import build_D
from utils import load_model
from obrs import density_ratio, find_c
from evaluate import get_classifier, embed

p = argparse.ArgumentParser()
p.add_argument('--G_name', default='G.pth')
p.add_argument('--D_name', default='D_ft1.pth')
p.add_argument('--D_arch', choices=['mlp', 'mlp_sn', 'cnn', 'cnn_sn'], default='mlp')
p.add_argument('--save_range', choices=['raw', '01t'], default='01t')
p.add_argument('--bg_thresh', type=float, default=0.2)
p.add_argument('--Ks', default='1,2,5,10,20,50')
p.add_argument('--n_pool', type=int, default=200_000)
p.add_argument('--per_row', type=int, default=16)
p.add_argument('--prefix', default='samples_vs_K')
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
G = load_model(Generator(g_output_dim=784).to(dev), 'checkpoints', dev, name=args.G_name).eval()
D = load_model(build_D(args.D_arch).to(dev), 'checkpoints', dev, name=args.D_name).eval()

xs, rs = [], []
with torch.no_grad():
    for _ in range(args.n_pool // 5000):
        x = G(torch.randn(5000, 100, device=dev))
        rs.append(density_ratio(D, x).cpu())
        xs.append(x.reshape(-1, 1, 28, 28).cpu())
x, r = torch.cat(xs), torch.cat(rs)
u = torch.rand(len(r))

def to_img(v):
    if args.save_range == 'raw':
        return v.clamp(0, 1)
    v = (v + 1) / 2
    return ((v - args.bg_thresh) / (1 - args.bg_thresh)).clamp(0, 1)

Ks = [float(k) for k in args.Ks.split(',')]
rows, labels, sets = [], [], []
for K in Ks:
    keep = torch.ones_like(r, dtype=torch.bool) if K == 1 else u < (find_c(r, K) * r).clamp(max=1)
    sets.append(keep)
    rows.append(x[keep][:args.per_row])
    labels.append(f'K = {K:g}  ({keep.float().mean():.1%} kept)')
top = r.topk(len(r) // 200).indices          # K -> infinity: the 0.5 % highest ratios
sets.append(torch.zeros_like(r, dtype=torch.bool).index_fill_(0, top, True))
rows.append(x[top][:args.per_row])
labels.append('top 0.5 % by r(x)')

os.makedirs('figures', exist_ok=True)
fig, axes = plt.subplots(len(rows), 1, figsize=(args.per_row * 0.7, len(rows) * 0.95))
for ax, row, lab in zip(axes, rows, labels):
    grid = torchvision.utils.make_grid(to_img(row), nrow=args.per_row, padding=2, pad_value=1)
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(f'figures/{args.prefix}_grid.png', dpi=150)

# Digit proportions of the accepted samples
clf = get_classifier(os.getenv('DATA') or 'data', dev)
pred = embed(clf, to_img(x), dev)[1]
fig, ax = plt.subplots(figsize=(8, 3.5))
width = 0.8 / len(sets)
for i, (keep, lab) in enumerate(zip(sets, labels)):
    share = torch.bincount(pred[keep], minlength=10).float() / keep.sum()
    ax.bar(torch.arange(10) + i * width, share.numpy(), width, label=lab)
ax.axhline(0.1, color='k', lw=0.8, ls='--')
ax.set_xticks(torch.arange(10) + 0.4 - width / 2, [str(d) for d in range(10)])
ax.set_xlabel('predicted digit'); ax.set_ylabel('share of accepted samples')
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f'figures/{args.prefix}_classes.png', dpi=150)
for keep, lab in zip(sets, labels):
    share = torch.bincount(pred[keep], minlength=10).float() / keep.sum()
    print(f'{lab:28s} ' + ' '.join(f'{s:.3f}' for s in share.tolist()))
print(f'saved figures/{args.prefix}_grid.png and figures/{args.prefix}_classes.png')
