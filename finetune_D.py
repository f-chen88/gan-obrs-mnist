"""Fine-tune the discriminator against the FROZEN final generator.

OBRS needs D(x) ~ p/(p + p_hat) for the final G. The D saved at the end of
training was chasing a moving G, so we retrain it a few epochs with G fixed
(BCE loss, same as training). Saves checkpoints/D_ft.pth (or --out_name).
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import Generator, Discriminator
from obrs import density_ratio
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune D for OBRS.')
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--G_name", type=str, default='G.pth')
    parser.add_argument("--out_name", type=str, default='D_ft.pth')
    parser.add_argument("--from_scratch", action='store_true',
                        help="Start from a new D instead of checkpoints/D.pth.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = os.getenv('DATA') or "data"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform,
                                   download=os.getenv('DATA') is None)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=4, pin_memory=True)

    mnist_dim = 784
    G = load_model(Generator(mnist_dim).to(device), 'checkpoints', device, name=args.G_name)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    D = Discriminator(mnist_dim).to(device)
    if not args.from_scratch and os.path.exists(os.path.join('checkpoints', 'D.pth')):
        D = load_model(D, 'checkpoints', device, name='D.pth')
        print('Starting from checkpoints/D.pth')
    else:
        print('Training D from scratch against the frozen G')

    criterion = nn.BCELoss()
    opt = optim.Adam(D.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        D.train()
        total = 0.0
        for x, _ in loader:
            x = x.view(-1, mnist_dim).to(device)
            with torch.no_grad():
                x_fake = G(torch.randn(x.shape[0], 100, device=device))
            out_real, out_fake = D(x), D(x_fake)
            loss = criterion(out_real, torch.ones_like(out_real)) \
                 + criterion(out_fake, torch.zeros_like(out_fake))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        # Diagnostic: E_{p_hat}[r] should be close to 1 for a calibrated D.
        D.eval()
        with torch.no_grad():
            r = density_ratio(D, G(torch.randn(10_000, 100, device=device)))
        print(f"epoch {epoch}: D loss {total / len(loader):.4f}, "
              f"mean r on fakes {r.mean():.3f}, median r {r.median():.3f}")

    torch.save(D.state_dict(), os.path.join('checkpoints', args.out_name))
    print(f'Saved checkpoints/{args.out_name}')
