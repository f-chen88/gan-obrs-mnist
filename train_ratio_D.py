"""Train a discriminator as a density-ratio estimator for OBRS, with G frozen.

Same protocol for every architecture, so that only the architecture differs:
- MNIST train set split into 50k (training) / 10k (validation, never trained on).
  The MNIST *test* set is left untouched (it is the evaluation reference).
- Loss: BCE (real = 1, fake = 0), fresh fakes from the frozen G at each step.
- Early stopping on the validation BCE (a proper scoring rule: it rewards
  well-calibrated probabilities, i.e. a good estimate of r = D/(1-D)).
  Validation fakes: a fixed set of 10k samples from G.
Usage:
  python train_ratio_D.py --arch cnn --out D_cnn_es.pth
  python train_ratio_D.py --arch mlp --init D.pth --out D_mlp_es.pth
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Generator
from d_models import build_D, n_params
from utils import load_model


@torch.no_grad()
def validate(D, val_real, val_fake, device, bs=2000):
    D.eval()
    p_r = torch.cat([D(val_real[i:i + bs].to(device)).squeeze(1) for i in range(0, len(val_real), bs)])
    p_f = torch.cat([D(val_fake[i:i + bs].to(device)).squeeze(1) for i in range(0, len(val_fake), bs)])
    eps = 1e-6
    p_r, p_f = p_r.clamp(eps, 1 - eps), p_f.clamp(eps, 1 - eps)
    bce = (-torch.log(p_r).mean() - torch.log(1 - p_f).mean()).item() / 2
    acc = ((p_r > .5).float().mean() + (p_f < .5).float().mean()).item() / 2
    mean_r = (p_f / (1 - p_f)).mean().item()
    D.train()
    return bce, acc, mean_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['mlp', 'cnn'], default='cnn')
    parser.add_argument('--init', default='', help='start from this checkpoint (e.g. D.pth, mlp only)')
    parser.add_argument('--G_name', default='G.pth')
    parser.add_argument('--out', required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=200, help='steps between validations')
    parser.add_argument('--patience', type=int, default=10, help='validations without improvement')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    data_path = os.getenv('DATA') or 'data'
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full = datasets.MNIST(data_path, train=True, transform=tf,
                          download=not os.path.exists(os.path.join(data_path, 'MNIST')))
    train_set, val_set = torch.utils.data.random_split(
        full, [50000, 10000], generator=torch.Generator().manual_seed(0))
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                         num_workers=4, drop_last=True)
    val_real = torch.stack([val_set[i][0] for i in range(len(val_set))]).view(-1, 784)

    G = load_model(Generator(784).to(device), 'checkpoints', device, name=args.G_name).eval()
    for p in G.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        gval = torch.Generator(device='cpu').manual_seed(1234)
        val_fake = torch.cat([G(torch.randn(2000, 100, generator=gval).to(device)).cpu()
                              for _ in range(5)])

    D = build_D(args.arch).to(device)
    if args.init:
        D = load_model(D, 'checkpoints', device, name=args.init)
    print(f'arch={args.arch}, params={n_params(D)}, init={args.init or "scratch"}')
    opt = torch.optim.Adam(D.parameters(), lr=args.lr)

    best = {'bce': float('inf')}
    log, step, bad = [], 0, 0
    bce, acc, mr = validate(D, val_real, val_fake, device)
    log.append({'step': 0, 'val_bce': bce, 'val_acc': acc, 'mean_r': mr})
    print(f'step 0: val BCE {bce:.4f}, acc {acc:.3f}, mean r {mr:.3f}')
    best = {'bce': bce, 'step': 0, 'state': {k: v.clone() for k, v in D.state_dict().items()}}

    stop = False
    for epoch in range(args.max_epochs):
        for x, _ in loader:
            x = x.view(-1, 784).to(device)
            with torch.no_grad():
                xf = G(torch.randn(x.shape[0], 100, device=device))
            out_r, out_f = D(x), D(xf)
            loss = F.binary_cross_entropy(out_r, torch.ones_like(out_r)) + \
                   F.binary_cross_entropy(out_f, torch.zeros_like(out_f))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % args.eval_every == 0:
                bce, acc, mr = validate(D, val_real, val_fake, device)
                log.append({'step': step, 'val_bce': bce, 'val_acc': acc, 'mean_r': mr})
                if bce < best['bce']:
                    best = {'bce': bce, 'step': step,
                            'state': {k: v.clone() for k, v in D.state_dict().items()}}
                    bad = 0
                else:
                    bad += 1
                print(f'step {step} (epoch {step * args.batch_size / 50000:.2f}): '
                      f'val BCE {bce:.4f}, acc {acc:.3f}, mean r {mr:.3f}'
                      f'{"  *best*" if bad == 0 else ""}', flush=True)
                if bad >= args.patience:
                    stop = True
                    break
        if stop:
            break

    D.load_state_dict(best['state'])
    bce, acc, mr = validate(D, val_real, val_fake, device)
    print(f'BEST at step {best["step"]} (epoch {best["step"] * args.batch_size / 50000:.2f}): '
          f'val BCE {bce:.4f}, acc {acc:.3f}, mean r {mr:.3f}')
    torch.save(D.state_dict(), os.path.join('checkpoints', args.out))
    json.dump({'arch': args.arch, 'init': args.init, 'params': n_params(D), 'best_step': best['step'],
               'best_epoch': best['step'] * args.batch_size / 50000, 'val_bce': bce, 'val_acc': acc,
               'mean_r': mr, 'curve': log},
              open(os.path.join('checkpoints', args.out.replace('.pth', '_trainlog.json')), 'w'), indent=1)
    print(f'Saved checkpoints/{args.out}')
