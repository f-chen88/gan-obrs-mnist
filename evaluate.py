import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms

# ---------- Configuration ----------
SEED = 42
SAMPLES_DIR = Path("samples")
N_REAL = None        # None = les 10 000 images du test set ; sinon ex. 1000
BATCH_SIZE = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Reproductibilité ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Même prétraitement pour réel et généré : 1 canal, 28x28, valeurs dans [0, 1]
tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def to_rgb(x):  # (B,1,28,28) -> (B,3,28,28)
    return x.repeat(1, 3, 1, 1)

# ---------- Images réelles : test set MNIST ----------
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)

if N_REAL is not None:
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(test_ds), generator=g)[:N_REAL].tolist()
    test_ds = Subset(test_ds, idx)

real_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Images générées : dossier samples/ ----------
class FolderImages(torch.utils.data.Dataset):
    def __init__(self, folder, transform):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.files = sorted(p for p in Path(folder).iterdir()
                            if p.suffix.lower() in exts)
        if not self.files:
            raise FileNotFoundError(f"Aucune image trouvée dans {folder}/")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("L"))

fake_ds = FolderImages(SAMPLES_DIR, tf)
fake_loader = DataLoader(fake_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Réelles : {len(test_ds)} images | Générées : {len(fake_ds)} images")

# ---------- Calcul du FID ----------
print('Loading FID metric...')
fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
print('FID loaded.')

with torch.no_grad():
    for x, _ in real_loader:
        fid.update(to_rgb(x).to(device), real=True)
        print('Real images processed')
    for x in fake_loader:
        fid.update(to_rgb(x).to(device), real=False)
        print('Generated images processed')

print(f"FID (seed={SEED}) : {fid.compute().item():.2f}")