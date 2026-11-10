"""Generate 10,000 MNIST samples with Optimal Budgeted Rejection Sampling (OBRS).

Default (final submission): generator trained with the OBRS loss (K_train = 5), spectrally normalised MLP
discriminator, EMA of G, snapshot of epoch 250; sampling with OBRS at K = 50 using the discriminator trained
with this generator. The generator architecture is the starter one (model.py).
"""
import torch
import torchvision
import os
import argparse
import json
import time

from model import Generator
from d_models import build_D
from utils import load_model
from obrs import calibrate, density_ratio, find_threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MNIST samples with OBRS.')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="The batch size to use for generation.")
    parser.add_argument("--K", type=float, default=50.0,
                        help="Sampling budget: average number of draws per kept sample (1 = no rejection).")
    parser.add_argument("--rule", choices=['obrs', 'threshold'], default='obrs',
                        help="obrs: accept with probability min(c_K r, 1). "
                             "threshold: keep the top 1/K by r (ablation of the report).")
    parser.add_argument("--G_name", type=str, default='search/obrs_K5_mlpsn_lrA_ema_s43/G_ep250.pth',
                        help="Generator checkpoint in checkpoints/.")
    parser.add_argument("--D_arch", choices=['mlp', 'mlp_sn', 'cnn', 'cnn_sn'], default='mlp_sn',
                        help="Architecture of the discriminator checkpoint (see d_models.py).")
    parser.add_argument("--D_name", type=str, default='search/obrs_K5_mlpsn_lrA_ema_s43/D_ep250.pth',
                        help="Discriminator checkpoint in checkpoints/.")
    parser.add_argument("--save_range", choices=['raw', '01', '01t'], default='01t',
                        help="raw: save G's tanh output as in the starter code (negative pixels -> black). "
                             "01: map [-1, 1] -> [0, 1] with (x+1)/2 (inverse of the training normalisation). "
                             "01t: as 01, then v -> max(0, (v - t)/(1 - t)) with t = --bg_thresh. "
                             "Only affects the saved PNGs, never what D sees.")
    parser.add_argument("--bg_thresh", type=float, default=0.0,
                        help="Background threshold t for --save_range 01t.")
    parser.add_argument("--gamma", type=float, default=1.25,
                        help="Contrast exponent applied to the saved intensities in [0, 1] (v -> v**gamma).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if os.path.exists('gen_log.json'):
        os.remove('gen_log.json')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    print('Model Loading...')
    mnist_dim = 784
    latent_dim = 100
    use_rejection = args.K > 1

    G = load_model(Generator(g_output_dim=mnist_dim).to(device), 'checkpoints', device, name=args.G_name).eval()
    D = None
    if use_rejection:
        D = load_model(build_D(args.D_arch).to(device), 'checkpoints', device, name=args.D_name).eval()
    print('Model loaded.')

    # Calibration: c_K such that the mean acceptance rate is 1/K, on 50,000 fresh samples.
    t0 = time.time()
    c, tau, mean_r = float('inf'), 0.0, None
    if use_rejection:
        c, r = calibrate(G, D, args.K, device, batch_size=args.batch_size)
        mean_r = r.mean().item()       # sanity check: close to 1 for a well-calibrated D
        if args.rule == 'threshold':
            tau = find_threshold(r, args.K)
            print(f'Threshold rejection: K={args.K}, tau={tau:.4g}, mean r on fakes={mean_r:.3f}')
        else:
            print(f'OBRS: K={args.K}, c_K={c:.4g}, mean r on fakes={mean_r:.3f}')
    calib_time = time.time() - t0

    def postprocess(x):
        if args.save_range in ('01', '01t'):
            x = (x + 1) / 2
        if args.save_range == '01t':
            x = ((x - args.bg_thresh) / (1 - args.bg_thresh)).clamp(min=0)
        if args.gamma != 1.0:
            x = x.clamp(0, 1) ** args.gamma
        return x

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    # With rejection, draw about K * batch_size latents per round (capped at 16384) so that each round
    # keeps about batch_size samples.
    draw_bs = min(16384, max(args.batch_size, int(args.batch_size * args.K))) if use_rejection else args.batch_size
    n_samples, n_drawn, n_accepted = 0, 0, 0
    while n_samples < 10000:
        z = torch.randn(draw_bs, latent_dim, device=device)
        n_drawn += draw_bs
        if use_rejection:              # OBRS keeps z with probability min(c_K * r(G(z)), 1)
            with torch.no_grad():
                r_batch = density_ratio(D, G(z))
            if args.rule == 'threshold':
                keep = r_batch >= tau
            else:
                a = torch.clamp(c * r_batch, max=1.0)
                keep = torch.rand_like(a) < a
            z = z[keep]
        n_accepted += z.shape[0]
        if z.shape[0] == 0:
            continue
        with torch.no_grad():
            x = postprocess(G(z).reshape(-1, 28, 28))
        for k in range(x.shape[0]):
            if n_samples < 10000:
                torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))
                n_samples += 1

    total_time = time.time() - t0
    accept_rate = n_accepted / n_drawn
    print(f'Done: {n_samples} samples saved; acceptance rate {accept_rate:.2%} '
          f'({n_accepted} accepted / {n_drawn} drawn).')
    print(f'Total generation time (calibration + sampling + saving): {total_time:.1f} s '
          f'(calibration {calib_time:.1f} s)')

    with open('gen_log.json', 'w') as f:
        json.dump({'G_name': args.G_name, 'D_name': args.D_name if D is not None else '',
                   'rule': args.rule if use_rejection else 'none', 'K': args.K,
                   'c_K': c if c != float('inf') else None, 'tau': tau, 'mean_r': mean_r,
                   'D_arch': args.D_arch, 'save_range': args.save_range,
                   'bg_thresh': args.bg_thresh if args.save_range == '01t' else None,
                   'gamma': args.gamma, 'seed': args.seed,
                   'n_drawn': n_drawn, 'n_accepted': n_accepted, 'accept_rate': round(accept_rate, 4),
                   'gen_time_s': round(total_time, 2), 'calib_time_s': round(calib_time, 2)},
                  f, indent=1)
