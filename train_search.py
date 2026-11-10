"""GAN training for the hyper-parameter search (vanilla or OBRS-trained G), with snapshots.

Same architecture, data pipeline and losses as train.py / utils.py. Differences:
  * separate learning rates for G and D, fixed seed, snapshots every --snap_every epochs;
  * --mode obrs trains G on the post-rejection distribution (Verine et al. 2024, sec. 4),
    with the loss of utils.G_train_obrs: mean(K a f(r / (K a))), f = Jensen-Shannon generator,
    a = min(c_K r, 1) (no gradient through a), c_K found on the batch;
  * --ratio logit computes r = exp(logit D) instead of D/(1-D) with D clamped to [eps, 1-eps]:
    the clamp kills the gradient of G when D saturates (the collapse seen with lr 2e-4).
  * --loss hinge / ls (vanilla mode) and --ema <decay> (EMA of G's weights, snapshots = EMA weights).
Outputs in checkpoints/<out>/: G_ep<e>.pth, D_ep<e>.pth, config.json, train_log.csv
    python train_search.py --mode obrs --K_train 5 --lrG 1e-4 --lrD 2.5e-5 --out search/obrs_K5
"""
import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import Generator
from d_models import build_D, d_logit, ARCHS, n_params

LOG2 = math.log(2.0)


def f_js(u):
    """Jensen-Shannon generator of the GAN (f(1) = 0), as in utils.f."""
    return u * torch.log(u) - (1 + u) * torch.log(1 + u) + 2 * LOG2


def ratio(D, x, kind, eps=1e-6):
    if kind == 'logit':
        return torch.exp(d_logit(D, x).clamp(-30, 30))
    d = D(x).squeeze(1).clamp(eps, 1 - eps)
    return d / (1 - d)


def find_c_fast(r, K, n_iter=60):
    """c such that mean(min(c r, 1)) = 1/K, by bisection on log c entirely on the device
    (same result as obrs.find_c, without one GPU synchronisation per bisection step)."""
    lo = torch.full((), -30.0, device=r.device)
    hi = torch.full((), 30.0, device=r.device)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        too_many = torch.clamp(torch.exp(mid) * r, max=1.0).mean() > 1.0 / K
        hi = torch.where(too_many, mid, hi)
        lo = torch.where(too_many, lo, mid)
    return torch.exp((lo + hi) / 2)


def D_step(x, G, D, opt, loss_kind, bce, dev):
    z = torch.randn(x.shape[0], 100, device=dev)
    with torch.no_grad():
        fake = G(z)
    if loss_kind == 'bce':
        out_r, out_f = D(x), D(fake)
        loss = bce(out_r, torch.ones_like(out_r)) + bce(out_f, torch.zeros_like(out_f))
    elif loss_kind == 'hinge':
        loss = torch.relu(1 - d_logit(D, x)).mean() + torch.relu(1 + d_logit(D, fake)).mean()
    else:  # least squares on the raw output (LSGAN, a = 0, b = c = 1)
        loss = ((d_logit(D, x) - 1) ** 2).mean() + (d_logit(D, fake) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.detach()


def G_step_vanilla(n, G, D, opt, loss_kind, bce, dev):
    fake = G(torch.randn(n, 100, device=dev))
    if loss_kind == 'bce':                       # non-saturating, as in the starter
        out = D(fake)
        loss = bce(out, torch.ones_like(out))
    elif loss_kind == 'hinge':
        loss = -d_logit(D, fake).mean()
    else:
        loss = ((d_logit(D, fake) - 1) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.detach(), None


def G_step_obrs(n, G, D, opt, K, kind, dev):
    r = ratio(D, G(torch.randn(n, 100, device=dev)), kind)
    rd = r.detach()
    rn = rd / rd.max()
    c = find_c_fast(rn, K)
    a = torch.clamp(c * rn, max=1.0)
    loss = torch.mean(a * K * f_js(r / (a * K)))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.detach(), a.mean().detach()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['vanilla', 'obrs'], default='vanilla')
    p.add_argument('--K_train', type=float, default=5.0)
    p.add_argument('--ratio', choices=['sigmoid', 'logit'], default='sigmoid')
    p.add_argument('--D_arch', choices=ARCHS, default='mlp')
    p.add_argument('--loss', choices=['bce', 'hinge', 'ls'], default='bce',
                   help='GAN loss for D and G (vanilla mode). OBRS mode always uses BCE for D.')
    p.add_argument('--ema', type=float, default=0.0,
                   help='EMA decay of G weights (e.g. 0.999); snapshots G_ep*.pth are then the EMA '
                        'weights and the raw ones are saved as Graw_ep*.pth')
    p.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 for G and D (DCGAN: 0.5)')
    p.add_argument('--lrG', type=float, default=1e-4)
    p.add_argument('--lrD', type=float, default=2.5e-5)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--snap_every', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', required=True, help='folder inside checkpoints/')
    p.add_argument('--init_dir', default='', help='resume from checkpoints/<init_dir>/ (fine-tuning)')
    p.add_argument('--init_ep', type=int, default=0, help='epoch of the snapshot to resume from')
    args = p.parse_args()
    if args.mode == 'obrs' and args.loss != 'bce':
        p.error('--mode obrs needs --loss bce (the OBRS loss uses D as a density-ratio estimator)')

    torch.manual_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join('checkpoints', args.out)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, 'config.json'), 'w'), indent=1)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data = datasets.MNIST(os.getenv('DATA') or 'data', train=True, transform=tf,
                          download=os.getenv('DATA') is None)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=4, pin_memory=True, drop_last=True)

    G, D = Generator(g_output_dim=784).to(dev), build_D(args.D_arch).to(dev)
    print(f'D: {args.D_arch}, {n_params(D):,} parameters')
    optG = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    bce = nn.BCELoss()
    if args.init_dir:   # fine-tuning: raw G weights if saved (EMA runs), its D, EMA started from the EMA weights
        src = os.path.join('checkpoints', args.init_dir)
        raw = os.path.join(src, f'Graw_ep{args.init_ep}.pth')
        G.load_state_dict(torch.load(raw if os.path.exists(raw) else os.path.join(src, f'G_ep{args.init_ep}.pth'),
                                     map_location=dev))
        D.load_state_dict(torch.load(os.path.join(src, f'D_ep{args.init_ep}.pth'), map_location=dev))
        print(f'resumed from {src} epoch {args.init_ep} ({"raw G" if os.path.exists(raw) else "G"})')
    G_ema = None
    if args.ema > 0:
        import copy
        G_ema = copy.deepcopy(G).eval()
        if args.init_dir:
            G_ema.load_state_dict(torch.load(os.path.join(src, f'G_ep{args.init_ep}.pth'), map_location=dev))
        for q in G_ema.parameters():
            q.requires_grad_(False)

    log = open(os.path.join(out, 'train_log.csv'), 'w', newline='')
    w = csv.writer(log)
    w.writerow(['epoch', 'D_loss', 'G_loss', 'accept_train', 'epoch_time_s'])
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0, dl, gl, acc, nb = time.time(), 0.0, 0.0, 0.0, 0
        G.train(); D.train()
        for x, _ in loader:
            x = x.view(-1, 784).to(dev, non_blocking=True)
            dl = dl + D_step(x, G, D, optD, args.loss, bce, dev)
            if args.mode == 'obrs':
                l, a = G_step_obrs(x.shape[0], G, D, optG, args.K_train, args.ratio, dev)
                acc = acc + a
            else:
                l, _ = G_step_vanilla(x.shape[0], G, D, optG, args.loss, bce, dev)
            gl = gl + l
            nb += 1
            if G_ema is not None:
                with torch.no_grad():
                    for pe, pg in zip(G_ema.parameters(), G.parameters()):
                        pe.lerp_(pg, 1 - args.ema)
        dl, gl, acc = float(dl), float(gl), float(acc)
        w.writerow([epoch, dl / nb, gl / nb, acc / nb if args.mode == 'obrs' else '',
                    round(time.time() - t0, 2)])
        log.flush()
        if not all(math.isfinite(v) for v in (dl, gl)):
            print(f'epoch {epoch}: non-finite loss, stopping')
            break
        if epoch % args.snap_every == 0 or epoch == args.epochs:
            if G_ema is not None:
                torch.save(G_ema.state_dict(), os.path.join(out, f'G_ep{epoch}.pth'))
                torch.save(G.state_dict(), os.path.join(out, f'Graw_ep{epoch}.pth'))
            else:
                torch.save(G.state_dict(), os.path.join(out, f'G_ep{epoch}.pth'))
            torch.save(D.state_dict(), os.path.join(out, f'D_ep{epoch}.pth'))
            print(f'epoch {epoch}: D {dl / nb:.4f}  G {gl / nb:.4f}  '
                  f'({time.time() - t_start:.0f} s elapsed)', flush=True)
    print(f'Training done in {time.time() - t_start:.0f} s')
