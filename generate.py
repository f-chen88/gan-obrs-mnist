import torch
import torchvision
import os
import argparse
import json
import time


from model import Generator, Discriminator
from d_models import build_D
from utils import load_model
import obrs
from obrs import calibrate, density_ratio, find_threshold
from dot import estimate_k_eff, dot_refine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MNIST samples (OBRS and/or DOT).')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for generation.")
    parser.add_argument("--K", type=float, default=50.0,
                      help="Rejection budget: average number of draws per kept sample (1 = no rejection).")
    parser.add_argument("--rule", choices=['obrs', 'threshold'], default='obrs',
                      help="obrs: accept with prob min(c_K r, 1) (optimal). "
                           "threshold: keep the top 1/K by r (ablation).")
    parser.add_argument("--calib", type=str, default='',
                      help="Calibration file from calibrate_D.py (e.g. checkpoints/D_ft_calib.json); "
                           "empty = raw D/(1-D).")
    parser.add_argument("--G_name", type=str, default='search/obrs_K5_mlpsn_lrA_ema_s43/G_ep250.pth',
                      help="Generator checkpoint in checkpoints/.")
    parser.add_argument("--D_arch", choices=['mlp', 'mlp_sn', 'cnn', 'cnn_sn'], default='mlp_sn',
                      help="Architecture of the discriminator checkpoint (see d_models.py).")
    parser.add_argument("--D_name", type=str, default='search/obrs_K5_mlpsn_lrA_ema_s43/D_ep250.pth',
                      help="Discriminator checkpoint in checkpoints/.")
    parser.add_argument("--dot_steps", type=int, default=0,
                      help="Latent DOT gradient steps (0 = no DOT).")
    parser.add_argument("--dot_lr", type=float, default=0.01,
                      help="Latent DOT step size (epsilon in the paper).")
    parser.add_argument("--k_eff", type=float, default=None,
                      help="Lipschitz constant of D o G for DOT (default: estimated).")
    parser.add_argument("--dot_naive", action='store_true',
                      help="Ablation: only climb D(G(z)), without the transport cost.")
    parser.add_argument("--save_range", choices=['raw', '01', '01t'], default='01t',
                      help="raw: save G's tanh output as in the starter code (negative pixels -> black). "
                           "01: map [-1,1] -> [0,1] with (x+1)/2 before saving (the inverse of the training "
                           "normalisation). 01t: as 01, then v -> max(0, (v - t)/(1 - t)) with t = --bg_thresh, "
                           "which turns the faint grey haze of the background back to black. "
                           "Only affects the saved PNGs, never what D sees.")
    parser.add_argument("--bg_thresh", type=float, default=0.0,
                      help="Background threshold t for --save_range 01t.")
    parser.add_argument("--z_temp", type=float, default=1.0,
                      help="Latent temperature: z ~ N(0, z_temp^2) (truncation-style trade-off; 1 = standard).")
    parser.add_argument("--gamma", type=float, default=1.25,
                      help="Gamma applied to the saved intensities in [0, 1] (v -> v**gamma), after --save_range.")
    parser.add_argument("--G_name2", type=str, default='',
                      help="Optional second generator: the samples are a mixture of the two (G, D) pairs.")
    parser.add_argument("--D_name2", type=str, default='',
                      help="Discriminator paired with --G_name2 (same --D_arch).")
    parser.add_argument("--mix", type=float, default=0.5,
                      help="Fraction of the 10000 samples drawn from the second pair.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if os.path.exists('gen_log.json'):
        os.remove('gen_log.json')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    latent_dim = 100

    use_rejection = args.K > 1
    use_dot = args.dot_steps > 0
    if use_dot and args.D_arch not in ('mlp', 'mlp_sn'):
        raise SystemExit('DOT is implemented for the MLP discriminators only.')

    # One (G, D) pair, or a mixture of two pairs (--G_name2/--D_name2): a fraction --mix of the
    # 10000 samples comes from the second pair, each pair with its own OBRS calibration.
    pairs = [(args.G_name, args.D_name)]
    if args.G_name2:
        pairs.append((args.G_name2, args.D_name2 or args.D_name))
    n2 = int(round(10000 * args.mix)) if len(pairs) == 2 else 0
    targets = [10000 - n2, n2] if len(pairs) == 2 else [10000]

    models = []
    for g_name, d_name in pairs:
        G = load_model(Generator(g_output_dim=mnist_dim).to(device), 'checkpoints', device, name=g_name).eval()
        D = None
        if use_rejection or use_dot:
            D = load_model(build_D(args.D_arch).to(device), 'checkpoints', device, name=d_name).eval()
        models.append([G, D])

    calib_a = None
    if args.calib:
        obrs.CALIB = json.load(open(args.calib))
        calib_a = obrs.CALIB['a']
        print(f"Using calibrated ratio: a={obrs.CALIB['a']:.4f}, b={obrs.CALIB['b']:.4f}")

    print('Model loaded.')

    t0 = time.time()
    c, tau, mean_r = float('inf'), 0.0, None
    for m in models:
        G, D = m
        c_m, tau_m, mean_r_m = float('inf'), 0.0, None
        if use_rejection:
            if args.z_temp == 1.0:
                c_m, r = calibrate(G, D, args.K, device, batch_size=args.batch_size)
            else:   # calibrate c_K on latents drawn at the same temperature as the generation
                with torch.no_grad():
                    r = torch.cat([density_ratio(D, G(torch.randn(args.batch_size, latent_dim, device=device) * args.z_temp))
                                   for _ in range(50_000 // args.batch_size + 1)])
                c_m = obrs.find_c(r, args.K)
            mean_r_m = r.mean().item()
            if args.rule == 'threshold':
                tau_m = find_threshold(r, args.K)
                print(f'Threshold rejection: K={args.K}, tau={tau_m:.4g}, mean r on fakes={mean_r_m:.3f}')
            else:
                # Sanity check: a well-trained D gives E_{p_hat}[r] close to 1.
                print(f'OBRS: K={args.K}, c_K={c_m:.4g}, mean r on fakes={mean_r_m:.3f}')
        m += [c_m, tau_m]
        if c == float('inf'):
            c, tau, mean_r = c_m, tau_m, mean_r_m
    calib_time = time.time() - t0

    k_eff = args.k_eff
    if use_dot:
        if k_eff is None:
            k_eff = estimate_k_eff(models[0][0], models[0][1], device, latent_dim=latent_dim)
        print(f'DOT: steps={args.dot_steps}, lr={args.dot_lr}, k_eff={k_eff:.4g}, '
              f'naive={args.dot_naive}')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples, n_drawn, n_accepted = 0, 0, 0
    # With rejection, draw about K * batch_size latents per round (capped at 16384) so that each round
    # keeps about batch_size samples: large matrix products are much faster than many small ones on CPU.
    draw_bs = min(16384, max(args.batch_size, int(args.batch_size * args.K))) if use_rejection else args.batch_size
    stop = 0
    for (G, D, c_m, tau_m), n_target in zip(models, targets):
      stop += n_target
      while n_samples < stop:
        z = torch.randn(draw_bs, latent_dim, device=device) * args.z_temp
        n_drawn += draw_bs

        # 1) Rejection: OBRS keeps z with probability min(c_K * r(G(z)), 1).
        if use_rejection:
            with torch.no_grad():
                r_batch = density_ratio(D, G(z))
            if args.rule == 'threshold':
                keep = r_batch >= tau_m
            else:
                a = torch.clamp(c_m * r_batch, max=1.0)
                keep = torch.rand_like(a) < a
            z = z[keep]
        n_accepted += z.shape[0]
        if z.shape[0] == 0:
            continue

        # 2) DOT: move each kept latent towards regions D prefers.
        if use_dot:
            z = dot_refine(G, D, z, k_eff, n_steps=args.dot_steps,
                           lr=args.dot_lr, naive=args.dot_naive)

        with torch.no_grad():
            x = G(z).reshape(-1, 28, 28)
        if args.save_range in ('01', '01t'):
            x = (x + 1) / 2
        if args.save_range == '01t':
            x = ((x - args.bg_thresh) / (1 - args.bg_thresh)).clamp(min=0)
        if args.gamma != 1.0:
            x = x.clamp(0, 1) ** args.gamma
        for k in range(x.shape[0]):
            if n_samples < stop:
                torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))
                n_samples += 1

    total_time = time.time() - t0
    accept_rate = n_accepted / n_drawn
    print(f'Done: {n_samples} samples saved; acceptance rate {accept_rate:.2%} '
          f'({n_accepted} accepted / {n_drawn} drawn).')
    print(f'Total generation time (calibration + sampling + saving): {total_time:.1f} s '
          f'(calibration {calib_time:.1f} s)')

    with open('gen_log.json', 'w') as f:
        json.dump({'G_name': args.G_name, 'D_name': args.D_name if models[0][1] is not None else '',
                   'G_name2': args.G_name2, 'mix': args.mix if args.G_name2 else 0,
                   'rule': args.rule if use_rejection else 'none', 'K': args.K,
                   'c_K': c if c != float('inf') else None, 'tau': tau,
                   'mean_r': mean_r, 'calib_a': calib_a, 'D_arch': args.D_arch, 'save_range': args.save_range,
                   'bg_thresh': args.bg_thresh if args.save_range == '01t' else None, 'dot_steps': args.dot_steps, 'z_temp': args.z_temp, 'gamma': args.gamma, 'seed': args.seed,
                   'n_drawn': n_drawn, 'n_accepted': n_accepted,
                   'accept_rate': round(accept_rate, 4),
                   'gen_time_s': round(total_time, 2), 'calib_time_s': round(calib_time, 2)},
                  f, indent=1)
