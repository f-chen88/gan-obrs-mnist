# DataLab Assignment 2, Group 15: Optimal Budgeted Rejection Sampling for a fixed MLP GAN

Franck CHEN, Olaf KWIATKOWSKI, Aissata GUINDO.

The generator architecture is the starter one (`model.py`) and is never changed. We improve its samples at
sampling time with **Optimal Budgeted Rejection Sampling** (OBRS, Verine et al., AISTATS 2024): a sample `x` is
kept with probability `min(c_K r(x), 1)`, where `r = D / (1 - D)` estimates the density ratio `p / p_hat` and
`c_K` is set by bisection so that one sample in `K` is kept. We then train the generator with the OBRS loss,
a spectrally normalised MLP discriminator and an EMA of the generator's weights.

## Generation (evaluated by the platform)

```bash
pip install -r requirements.txt
python generate.py --batch_size 256      # writes 10,000 PNGs to samples/ and gen_log.json
```

Default configuration (final submission): `checkpoints/search/obrs_K5_mlpsn_lrA_ema_s43/G_ep250.pth` and the
discriminator trained with it (`D_ep250.pth`, spectrally normalised MLP), OBRS with `K = 50`, images written as
`max((x+1)/2, 0) ** 1.25`. Useful options: `--K 1` (no rejection), `--rule threshold` (deterministic top-1/K
ablation), `--G_name G.pth --D_name D_ft1.pth --D_arch mlp --save_range raw --K 10` (OBRS on the frozen
100-epoch baseline, push 1).

## Files

| File | Role |
|---|---|
| `generate.py` | sampling with OBRS (run by the platform) |
| `obrs.py` | density ratio, bisection for `c_K`, calibration, threshold ablation |
| `model.py`, `utils.py`, `train.py` | starter generator, discriminator and training |
| `d_models.py` | discriminators: starter MLP, MLP with spectral normalisation, CNNs |
| `train_search.py` | training with the vanilla or OBRS loss, spectral norm, EMA, snapshots every 25 epochs |
| `finetune_D.py`, `train_ratio_D.py` | retrain a discriminator against a frozen generator (ratio estimators of the report) |
| `evaluate.py` | local FID (Inception), precision, recall, digit balance `TV_cls` |
| `samples_vs_K.py` | grid of accepted samples and digit frequencies against `K` (report, Fig. 2) |
| `search/` | Slurm pipeline of the training search: configurations, training, snapshot evaluation, generation tuning, summary |
| `scripts/` | Slurm launchers for training and generation on Juliet |

## Reproducing the training of the final model

```bash
python train_search.py --mode obrs --K_train 5 --ratio logit --D_arch mlp_sn \
    --lrG 2e-4 --lrD 2e-4 --beta1 0.5 --ema 0.999 --seed 43 --out search/obrs_K5_mlpsn_lrA_ema_s43
```

300 epochs, batch 64, snapshot of epoch 250 used. The full search is launched with `bash search/launch.sh`
(configurations in `search/configs*.txt`).

## Results

| Submission | Platform FID | Local FID |
|---|---|---|
| Baseline (starter, 100 epochs) | 29.82 | 29.56 |
| OBRS on the frozen baseline, `K = 10` | 25.18 | 24.75 |
| OBRS-trained G, spectral norm, EMA, OBRS `K = 50` (final) | see report | 5.97 |
