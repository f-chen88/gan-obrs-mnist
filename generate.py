import torch 
import torchvision
import os
import argparse
import time


from model import Generator, Discriminator
from utils import load_model

stabilize_constant = 1e-8

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Samples from given model.')
    parser.add_argument("--model-path", type=str, default="checkpoints",
                        help="Path to the directory containing the trained model checkpoints.")
    parser.add_argument("--method", type=str, default='vanilla', choices=['vanilla', 'obrs'],
                        help="The method to use for generating samples.")
    #parser.add_argument("--run-dir", type=str,)
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for generation.")
    parser.add_argument("--budget", type=float, default=2.5, 
                        help="The budget for OBRS. Only used if --method is compatible with OBRS.")
    parser.add_argument("--precision", type=float, default=1e-5,
                        help="The precision for OBRS dichotomic search.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)



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

    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator = load_model(generator , args.model_path, 'G', device)
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
    generator.eval()

    if args.method in ['obrs']:
        discriminator = Discriminator(d_input_dim = mnist_dim).to(device)
        discriminator = load_model(discriminator , args.model_path, 'D', device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(discriminator)
    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    start = time.time()
    n_samples = 0

    if args.method == 'vanilla':
        with torch.no_grad():
            while n_samples<10000:
                z = torch.randn(args.batch_size, 100).to(device)
                x = generator(z)
                x = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples<10000:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                        n_samples += 1
        print (time.time() - start)

    if args.method == 'obrs':
        from utils import search_optimal_threshold
        with torch.no_grad():
            while n_samples<10000:
                z = torch.randn(args.batch_size, 100).to(device)
                x = generator(z)
                d_output = discriminator(x).squeeze()
                likelihood_ratios = d_output / (1 + stabilize_constant - d_output)
                max_likelihood_ratio = torch.max(likelihood_ratios)
                likelihood_ratios = likelihood_ratios / max_likelihood_ratio
                optimal_threshold = search_optimal_threshold(likelihood_ratios, args.budget, args.precision)
                acceptance_rates = torch.clamp(likelihood_ratios * optimal_threshold, max=1.0)
                acceptance_mask = torch.bernoulli(acceptance_rates).bool()
                x = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples<10000 and acceptance_mask[k]:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                        n_samples += 1
        print (f"Execution time {time.time() - start}")