"""Local evaluation of samples/ against MNIST: FID, precision, recall.

The course platform's exact feature extractor is not published, so absolute
numbers here will differ from the platform's. Use this script to COMPARE
configurations (same features, same reference set), not to predict the grade.

Features: penultimate layer (128-d) of a small CNN trained on MNIST
(trained once and cached in eval_classifier.pth).
- FID: Frechet distance between Gaussian fits of real and generated features.
- Precision / recall: k-NN support estimates (Kynkaanniemi et al., 2019),
  i.e. the "precision and recall for finite support" estimator of the course,
  computed on L2-normalised features. Without normalisation, out-of-distribution
  images (e.g. noise) land near the origin, inside the huge k-NN balls of a few
  atypical real digits, and get a precision of ~0.5 (we measured it).
- Density / coverage (Naeem et al., 2020): outlier-robust variants of P / R.

Usage:  python evaluate.py --label "K=2"      (appends a row to results.csv)
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import datasets, transforms


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return F.relu(self.fc1(x.flatten(1)))

    def forward(self, x):
        return self.fc2(self.features(x))


def get_classifier(data_path, device, path='eval_classifier.pth', epochs=3):
    clf = Classifier().to(device)
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()
    print('Training the evaluation classifier (once)...')
    ds = datasets.MNIST(data_path, train=True, download=not os.path.exists(
        os.path.join(data_path, 'MNIST')), transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    torch.manual_seed(0)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(clf.state_dict(), path)
    return clf.eval()


@torch.no_grad()
def embed(clf, images, device, bs=1000):
    feats, preds = [], []
    for i in range(0, len(images), bs):
        x = images[i:i + bs].to(device)
        f = clf.features(x)
        feats.append(f.cpu())
        preds.append(clf.fc2(f).argmax(1).cpu())
    return torch.cat(feats), torch.cat(preds)


def load_samples(folder):
    files = sorted(os.listdir(folder), key=lambda s: int(os.path.splitext(s)[0]))
    imgs = [np.array(Image.open(os.path.join(folder, f)).convert('L'), dtype=np.float32) / 255.
            for f in files]
    return torch.from_numpy(np.stack(imgs))[:, None]


def fid(a, b):
    a, b = a.double().numpy(), b.double().numpy()
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    eps = 1e-6 * np.eye(len(mu1))      # dead ReLU units make covariances singular
    s1, s2 = s1 + eps, s2 + eps
    covmean = linalg.sqrtm(s1 @ s2)
    if isinstance(covmean, tuple):      # older SciPy may return (sqrtm, error)
        covmean = covmean[0]
    covmean = np.real(covmean)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def knn_radii(x, k, bs=2000):
    r = []
    for i in range(0, len(x), bs):
        d = torch.cdist(x[i:i + bs], x)
        r.append(d.kthvalue(k + 1, dim=1).values)   # k+1: skip the point itself
    return torch.cat(r)


def density_coverage(fake, real, k=5, bs=2000):
    radii = knn_radii(real, k)
    density, covered = 0.0, torch.zeros(len(real), dtype=torch.bool)
    for i in range(0, len(fake), bs):
        inside = torch.cdist(fake[i:i + bs], real) <= radii[None, :]
        density += inside.sum().item()
        covered |= inside.any(0)
    return density / (k * len(fake)), covered.float().mean().item()


def coverage_rate(queries, refs, radii, bs=2000):
    """Fraction of queries inside at least one k-NN ball of refs."""
    inside = []
    for i in range(0, len(queries), bs):
        d = torch.cdist(queries[i:i + bs], refs)
        inside.append((d <= radii[None, :]).any(1))
    return torch.cat(inside).float().mean().item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default='samples')
    parser.add_argument('--label', default='')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--out', default='results.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = os.getenv('DATA') or 'data'
    t0 = time.time()

    clf = get_classifier(data_path, device)
    real = datasets.MNIST(data_path, train=False, download=False,
                          transform=transforms.ToTensor())
    real_x = torch.stack([real[i][0] for i in range(len(real))])
    fake_x = load_samples(args.samples)

    real_f, real_pred = embed(clf, real_x, device)
    fake_f, fake_pred = embed(clf, fake_x, device)

    real_n, fake_n = F.normalize(real_f, dim=1), F.normalize(fake_f, dim=1)
    dens, cov = density_coverage(fake_n, real_n)
    res = {
        'label': args.label,
        'n_samples': len(fake_x),
        'FID': round(fid(real_f, fake_f), 3),
        'precision': round(coverage_rate(fake_n, real_n, knn_radii(real_n, args.k)), 4),
        'recall': round(coverage_rate(real_n, fake_n, knn_radii(fake_n, args.k)), 4),
        'density': round(dens, 4),
        'coverage': round(cov, 4),
    }
    counts = torch.bincount(fake_pred, minlength=10).tolist()
    res['min_class_share'] = round(min(counts) / len(fake_pred), 4)
    res['class_counts'] = ' '.join(map(str, counts))
    # Total-variation distance between generated and real digit proportions (0 = identical)
    p_fake = torch.bincount(fake_pred, minlength=10).float() / len(fake_pred)
    p_real = torch.bincount(real_pred, minlength=10).float() / len(real_pred)
    res['class_TV'] = round(0.5 * (p_fake - p_real).abs().sum().item(), 4)

    # Generation info written by generate.py (acceptance rate, time, ...)
    if os.path.exists('gen_log.json') and os.path.abspath(args.samples) == os.path.abspath('samples'):
        log = json.load(open('gen_log.json'))
        for key in ['rule', 'K', 'accept_rate', 'gen_time_s', 'calib_time_s', 'mean_r', 'calib_a', 'D_arch', 'save_range',
                    'dot_steps', 'seed', 'G_name', 'D_name']:
            res[key] = log.get(key)

    for key, v in res.items():
        print(f'{key:>16}: {v}')
    print(f'(evaluation time {time.time() - t0:.0f} s)')

    # Append, merging columns with any existing results file
    rows, fields = [], []
    if os.path.exists(args.out):
        with open(args.out, newline='') as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), list(reader.fieldnames or [])
    fields += [k for k in res if k not in fields]
    rows.append({k: res.get(k, '') for k in res})
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader()
        w.writerows(rows)
