import torch
import torchvision
import os
import argparse
import json
import time


from model import Generator, Discriminator
from utils import load_model
import obrs
from obrs import calibrate, density_ratio, find_threshold
from dot import estimate_k_eff, dot_refine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MNIST samples (OBRS and/or DOT).')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for generation.")
    parser.add_argument("--K", type=float, default=2.0,
                      help="Rejection budget: average number of draws per kept sample (1 = no rejection).")
    parser.add_argument("--rule", choices=['obrs', 'threshold'], default='obrs',
                      help="obrs: accept with prob min(c_K r, 1) (optimal). "
                           "threshold: keep the top 1/K by r (ablation).")
    parser.add_argument("--calib", type=str, default='',
                      help="Calibration file from calibrate_D.py (e.g. checkpoints/D_ft_calib.json); "
                           "empty = raw D/(1-D).")
    parser.add_argument("--G_name", type=str, default='G.pth',
                      help="Generator checkpoint in checkpoints/.")
    parser.add_argument("--D_name", type=str, default='D_ft1.pth',
                      help="Discriminator checkpoint in checkpoints/.")
    parser.add_argument("--dot_steps", type=int, default=0,
                      help="Latent DOT gradient steps (0 = no DOT).")
    parser.add_argument("--dot_lr", type=float, default=0.01,
                      help="Latent DOT step size (epsilon in the paper).")
    parser.add_argument("--k_eff", type=float, default=None,
                      help="Lipschitz constant of D o G for DOT (default: estimated).")
    parser.add_argument("--dot_naive", action='store_true',
                      help="Ablation: only climb D(G(z)), without the transport cost.")
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

    G = Generator(g_output_dim=mnist_dim).to(device)
    G = load_model(G, 'checkpoints', device, name=args.G_name)
    G.eval()

    use_rejection = args.K > 1
    use_dot = args.dot_steps > 0
    D = None
    if use_rejection or use_dot:
        D = Discriminator(mnist_dim).to(device)
        D = load_model(D, 'checkpoints', device, name=args.D_name)
        D.eval()

    calib_a = None
    if args.calib:
        obrs.CALIB = json.load(open(args.calib))
        calib_a = obrs.CALIB['a']
        print(f"Using calibrated ratio: a={obrs.CALIB['a']:.4f}, b={obrs.CALIB['b']:.4f}")

    if args.method in ['obrs']:
        discriminator = Discriminator(d_input_dim = mnist_dim).to(device)
        discriminator = load_model(discriminator , args.model_path, 'D', device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(discriminator)
    print('Model loaded.')

    t0 = time.time()
    c, tau, mean_r = float('inf'), 0.0, None
    if use_rejection:
        c, r = calibrate(G, D, args.K, device, batch_size=args.batch_size)
        mean_r = r.mean().item()
        if args.rule == 'threshold':
            tau = find_threshold(r, args.K)
            print(f'Threshold rejection: K={args.K}, tau={tau:.4g}, mean r on fakes={mean_r:.3f}')
        else:
            # Sanity check: a well-trained D gives E_{p_hat}[r] close to 1.
            print(f'OBRS: K={args.K}, c_K={c:.4g}, mean r on fakes={mean_r:.3f}')
    calib_time = time.time() - t0

    k_eff = args.k_eff
    if use_dot:
        if k_eff is None:
            k_eff = estimate_k_eff(G, D, device, latent_dim=latent_dim)
        print(f'DOT: steps={args.dot_steps}, lr={args.dot_lr}, k_eff={k_eff:.4g}, '
              f'naive={args.dot_naive}')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples, n_drawn, n_accepted = 0, 0, 0
    while n_samples < 10000:
        z = torch.randn(args.batch_size, latent_dim, device=device)
        n_drawn += args.batch_size

        # 1) Rejection: OBRS keeps z with probability min(c_K * r(G(z)), 1).
        if use_rejection:
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

        # 2) DOT: move each kept latent towards regions D prefers.
        if use_dot:
            z = dot_refine(G, D, z, k_eff, n_steps=args.dot_steps,
                           lr=args.dot_lr, naive=args.dot_naive)

        with torch.no_grad():
            x = G(z).reshape(-1, 28, 28)
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
                   'c_K': c if c != float('inf') else None, 'tau': tau,
                   'mean_r': mean_r, 'calib_a': calib_a, 'dot_steps': args.dot_steps, 'seed': args.seed,
                   'n_drawn': n_drawn, 'n_accepted': n_accepted,
                   'accept_rate': round(accept_rate, 4),
                   'gen_time_s': round(total_time, 2), 'calib_time_s': round(calib_time, 2)},
                  f, indent=1)
