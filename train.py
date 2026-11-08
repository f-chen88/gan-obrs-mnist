import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from utils import D_train, G_train, save_models, G_train_obrs, load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    parser.add_argument("--checkpoints-path", type=str, default="checkpoints", help="path of the folder where the model will be saved")
    parser.add_argument("--obrs", action='store_true', help="Use OBRS for generator training.")
    parser.add_argument("--budget", type=float, default=2.0, help="Budget for OBRS (only used if --obrs is set).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--G_name", type=str, default='G.pth',
                      help="Generator checkpoint in checkpoints/.")
    parser.add_argument("--D_name", type=str, default='D.pth',
                      help="Discriminator checkpoint in checkpoints/.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    to_download=False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"Using device: CUDA")
        # Use all available GPUs if args.gpus is -1
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"Using device: CPU")
        

    

    # Create directories
    os.makedirs(args.checkpoints_path, exist_ok=True)
    data_path = os.getenv('DATA')
    if data_path is None:
        data_path = "data"
        to_download = True
    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=to_download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print('Dataset loaded.')

    # Model setup
    print('Model loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    if args.resume:
        G = load_model(G, args.checkpoints_path, device, name=args.G_name)
        D = load_model(D, args.checkpoints_path, device, name=args.D_name)

    # Wrap models in DataParallel if multiple GPUs are available
    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')

    # Loss and optimizers
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr/4 )


    print('Start training:')
    n_epoch = args.epochs
    if not args.obrs:
        for epoch in range(1, n_epoch + 1):
            D_loss, G_loss = 0 , 0
            print('=' * 30 + f'Epoch {epoch}' + '=' * 30)
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim).to(device)
                D_loss += D_train(x, G, D, D_optimizer, criterion, device)
                G_loss += G_train(x, G, D, G_optimizer, criterion, device)
                if batch_idx % 100 == 0:
                    print(f"Discriminator loss: {D_loss:.2f}")
                    print(f"Generator loss: {G_loss:.2f}")
                    D_loss = 0
                    G_loss = 0

            if epoch % 10 == 0:
                save_models(G, D, args.checkpoints_path)

    if args.obrs:
        print('Training with OBRS:')
        for epoch in range(1, n_epoch + 1):
            D_loss = 0
            G_loss = 0
            print('=' * 30 + f'Epoch {epoch}' + '=' * 30)
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim).to(device)
                D_loss += D_train(x, G, D, D_optimizer, criterion, device)
                G_loss_add, r, acceptance_ratios = G_train_obrs(x, G, D, G_optimizer, args.budget, device)
                G_loss += G_loss_add
                if batch_idx % 100 == 0:
                    print(f"Discriminator loss: {D_loss:.2f}")
                    print(f"Generator loss: {G_loss:.2f}")
                    D_loss = 0
                    G_loss = 0
                    print(f"mean r: {r.mean().item()}")
                    print(f"median r: {r.median().item()}")
                    print(f"quantile 0.99: {r.quantile(0.99)}")
                    print(f"quantile 0.01: {r.quantile(0.01)}")
                    #print(r)
                    #print(acceptance_ratios)

            if epoch % 2 == 0:
                save_models(G, D, args.checkpoints_path)

    print('Training done.')
