"""Latent-space Discriminator Optimal Transport (Tanaka, NeurIPS 2019, Algorithm 2).

For each latent z_y ~ N(0, I), we search a nearby z that the discriminator prefers:

    z* = argmin_z  ||z - z_y + delta||_2  -  D(G(z)) / k_eff

D(.) is the discriminator *logit* (before the sigmoid), as in the paper where
D: X -> R. k_eff is the empirical Lipschitz constant of D o G (eq. 21).
The 'naive' variant drops the distance term and just climbs D(G(z)) (eq. 56).
"""
import torch
import torch.nn.functional as F


def d_logit(D, x):
    """Discriminator output *before* the final sigmoid (same layers as model.py)."""
    x = F.leaky_relu(D.fc1(x), 0.2)
    x = F.leaky_relu(D.fc2(x), 0.2)
    x = F.leaky_relu(D.fc3(x), 0.2)
    return D.fc4(x).squeeze(1)


@torch.no_grad()
def estimate_k_eff(G, D, device, n_pairs=100, n_trials=10, latent_dim=100):
    """k_eff = max |D(G(z1)) - D(G(z2))| / ||z1 - z2|| over random pairs (eq. 21),
    averaged over n_trials independent draws, as in the paper's appendix."""
    vals = []
    for _ in range(n_trials):
        z1 = torch.randn(n_pairs, latent_dim, device=device)
        z2 = torch.randn(n_pairs, latent_dim, device=device)
        num = (d_logit(D, G(z1)) - d_logit(D, G(z2))).abs()
        den = (z1 - z2).norm(dim=1)
        vals.append((num / den).max().item())
    return sum(vals) / len(vals)


def dot_refine(G, D, z_y, k_eff, n_steps=20, lr=0.01, delta=1e-3, naive=False):
    """Run n_steps of projected gradient descent on z (Algorithm 2, SGD).

    The objective is a sum of independent per-sample terms, so one backward
    pass on the batch gives every sample its own gradient.
    """
    for p in list(G.parameters()) + list(D.parameters()):
        p.requires_grad_(False)

    dim = z_y.shape[1]
    z = z_y.clone().requires_grad_(True)
    for _ in range(n_steps):
        score = d_logit(D, G(z)) / k_eff
        if naive:
            obj = -score
        else:
            obj = (z - z_y + delta).norm(dim=1) - score
        g, = torch.autograd.grad(obj.sum(), z)
        with torch.no_grad():
            # Keep z on the sphere of radius ~sqrt(dim) where N(0, I) lives:
            # remove the radial component of the gradient (tangent projection).
            # The paper writes g - (g.z) z / sqrt(D); we use the exact tangent
            # projection g - (g.z) z / ||z||^2, which is what that step approximates.
            radial = (g * z).sum(dim=1, keepdim=True) / (z * z).sum(dim=1, keepdim=True)
            g = g - radial * z
            z -= lr * g
    return z.detach()
