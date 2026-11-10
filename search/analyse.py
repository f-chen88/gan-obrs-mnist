"""Helpers of the hyper-parameter search (run from the repository root).
    python search/analyse.py select --n 4          # rank snapshots -> results_search/candidates.txt
    python search/analyse.py pick --csv X --step K --key K   # best value of a key within a step
    python search/analyse.py summary               # final tables, figure, recommended push
Labels look like "run=van_lrB;ep=150;D=ft1;K=10;t=0.2;rule=obrs;step=snap | seed 0".
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict

RES = os.environ.get('RES', 'results_search')


def parse(label):
    cfg, _, seed = label.partition(' | seed ')
    d = dict(kv.split('=', 1) for kv in cfg.split(';'))
    d['seed'] = seed.strip()
    return d


def rows(pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(RES, pattern))):
        for r in csv.DictReader(open(path)):
            if r.get('FID_incep') in (None, ''):
                continue
            r.update(parse(r['label']))
            r['FID_incep'] = float(r['FID_incep'])
            out.append(r)
    return out


def mean_sd(v):
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0)


def fnum(r, key):
    try:
        return float(r[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def snap_table():
    tab = defaultdict(dict)   # (run, ep) -> {K: row}
    for r in rows('snap_*.csv'):
        tab[(r['run'], int(r['ep']))][r['K']] = r
    return tab


def select(n):
    tab = snap_table()
    scored = []
    for (run, ep), byK in tab.items():
        fids = [byK[k]['FID_incep'] for k in ('1', '10') if k in byK]
        if fids:
            scored.append((min(fids), run, ep))
    scored.sort()
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, 'candidates.txt'), 'w') as f:
        for _, run, ep in scored[:n]:
            f.write(f'{run} {ep}\n')
    print(f'{len(scored)} snapshots evaluated; best {n}:')
    for s, run, ep in scored[:n]:
        print(f'  {run:28s} epoch {ep:3d}  best FID {s:.2f}')


def pick(path, step, key):
    groups = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get('FID_incep') in (None, ''):
            continue
        d = parse(r['label'])
        if d.get('step') == step:
            groups[d[key]].append(float(r['FID_incep']))
    best = min(groups, key=lambda k: st.mean(groups[k]))
    print(best)


def summary():
    os.makedirs('figures', exist_ok=True)
    # 1. Training cost
    print('== Training runs (300 epochs, snapshots every 25)')
    print(f"{'run':28s} {'train time (min)':>17s}  options")
    mine = {r['run'] for r in rows('snap_*.csv')}
    for cfgp in sorted(glob.glob('checkpoints/search/*/config.json')):
        run = os.path.basename(os.path.dirname(cfgp))
        if mine and run not in mine:
            continue
        logp = os.path.join(os.path.dirname(cfgp), 'train_log.csv')
        t = sum(float(r['epoch_time_s']) for r in csv.DictReader(open(logp))) / 60 if os.path.exists(logp) else float('nan')
        c = json.load(open(cfgp))
        opts = f"{c['mode']} K={c['K_train'] if c['mode'] == 'obrs' else '-'} D={c.get('D_arch', 'mlp')} " \
               f"lrG={c['lrG']} lrD={c['lrD']} beta1={c.get('beta1', 0.9)} " \
               f"ratio={c['ratio'] if c['mode'] == 'obrs' else '-'} bs={c['batch_size']} seed={c['seed']}"
        print(f'{run:28s} {t:17.1f}  {opts}')

    # 2. Snapshots: FID vs epoch, without rejection and with OBRS K=10 (t = 0.2, seed 0)
    tab = snap_table()
    runs = sorted({run for run, _ in tab})
    with open(os.path.join(RES, 'snap_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'epoch', 'FID_K1', 'FID_K10', 'P_K10', 'R_K10', 'classTV_K1', 'classTV_K10'])
        for (run, ep) in sorted(tab):
            b = tab[(run, ep)]
            g = lambda k, c: fnum(b[k], c) if k in b else float('nan')
            w.writerow([run, ep, g('1', 'FID_incep'), g('10', 'FID_incep'), g('10', 'precision_incep'),
                        g('10', 'recall_incep'), g('1', 'class_TV'), g('10', 'class_TV')])
    print('\n== Snapshots (t = 0.2, seed 0): best epoch per run')
    print(f"{'run':28s} {'best K=1 (ep)':>16s} {'best K=10 (ep)':>16s} {'K=1 @300':>9s} {'K=10 @300':>10s}")
    for run in runs:
        eps = sorted(ep for r, ep in tab if r == run)
        def best(k):
            v = [(tab[(run, e)][k]['FID_incep'], e) for e in eps if k in tab[(run, e)]]
            return min(v) if v else (float('nan'), 0)
        b1, b10 = best('1'), best('10')
        last = tab.get((run, 300), {})
        l1 = last['1']['FID_incep'] if '1' in last else float('nan')
        l10 = last['10']['FID_incep'] if '10' in last else float('nan')
        print(f'{run:28s} {b1[0]:9.2f} ({b1[1]:3d}) {b10[0]:9.2f} ({b10[1]:3d}) {l1:9.2f} {l10:10.2f}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, k, title in zip(axes, ('1', '10'), ('no rejection (K = 1)', 'OBRS K = 10')):
            for run in runs:
                pts = sorted((ep, tab[(r, ep)][k]['FID_incep']) for r, ep in tab if r == run and k in tab[(r, ep)])
                if pts:
                    ax.plot(*zip(*pts), marker='o', ms=3, ls='-' if run.startswith('obrs') else '--', label=run)
            ax.set_title(f'FID Inception vs epoch, t = 0.2, {title}')
            ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
        axes[0].set_ylabel('FID (Inception, local)')
        axes[1].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(f'figures/{RES}_fid_vs_epoch.png', dpi=150)
        print(f'figure: figures/{RES}_fid_vs_epoch.png')
    except ImportError:
        print('(matplotlib not installed: no figure)')

    # 3. Generation-side search per candidate
    gen = rows('gen_*.csv')
    if not gen:
        print('\nNo generation-side results yet.')
        return
    print('\n== Generation-side search (seed 0 unless "final")')
    cands = sorted({(r['run'], r['ep']) for r in gen})
    finals = []
    for run, ep in cands:
        rs = [r for r in gen if r['run'] == run and r['ep'] == ep]
        print(f'\n-- {run} epoch {ep}')
        for step, key in (('D', 'D'), ('K', 'K'), ('t', 't'), ('rule', 'rule')):
            line = [f"{r[key]}: {r['FID_incep']:.2f}" for r in rs if r['step'] == step]
            if line:
                print(f'   {step:5s} ' + ' | '.join(line))
        fin = [r for r in rs if r['step'] == 'final']
        if fin:
            m, s = mean_sd([r['FID_incep'] for r in fin])
            P = st.mean(fnum(r, 'precision_incep') for r in fin)
            R = st.mean(fnum(r, 'recall_incep') for r in fin)
            tv = st.mean(fnum(r, 'class_TV') for r in fin)
            acc = st.mean(fnum(r, 'accept_rate') for r in fin)
            r0 = fin[0]
            print(f"   FINAL D={r0['D']} K={r0['K']} t={r0['t']}: FID {m:.2f} ± {s:.2f}  P {P:.3f}  R {R:.3f}  "
                  f"class_TV {tv:.3f}  acceptance {acc:.3f}")
            finals.append((m, s, run, ep, r0['D'], r0['K'], r0['t']))
    if finals:
        finals.sort()
        m, s, run, ep, D, K, t = finals[0]
        dfile = {'co': f'D_ep{ep}.pth', 'ft1': f'Dft1_ep{ep}.pth', 'ft3': f'Dft3_ep{ep}.pth'}[D]
        print('\n== Ranking of the candidates (3 seeds)')
        for f_ in finals:
            print(f'   {f_[0]:6.2f} ± {f_[1]:.2f}  {f_[2]} epoch {f_[3]}  D={f_[4]} K={f_[5]} t={f_[6]}')
        print(f'\n== RECOMMENDED PUSH: {run} epoch {ep}, D={D}, K={K}, t={t} -> local FID {m:.2f} ± {s:.2f}')
        print(f'   files : checkpoints/search/{run}/G_ep{ep}.pth  and  checkpoints/search/{run}/{dfile}')
        arch = json.load(open(f'checkpoints/search/{run}/config.json')).get('D_arch', 'mlp')
        print(f'   generate.py defaults: --D_arch {arch} --G_name search/{run}/G_ep{ep}.pth --D_name search/{run}/{dfile} '
              f'--K {K} --save_range 01t --bg_thresh {t}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['select', 'pick', 'summary'])
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--csv')
    p.add_argument('--step')
    p.add_argument('--key')
    a = p.parse_args()
    if a.cmd == 'select':
        select(a.n)
    elif a.cmd == 'pick':
        pick(a.csv, a.step, a.key)
    else:
        summary()
