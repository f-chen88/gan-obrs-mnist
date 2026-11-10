"""Optimal Budgeted Rejection Sampling (Verine et al., AISTATS 2024).

Acceptance rule (Theorem 3.1):  a(x) = min(c_K * r(x), 1)
  - r(x) ~ p(x) / p_hat(x) is estimated with the discriminator,
  - c_K is chosen so that the mean acceptance rate is exactly 1/K
    (K = average number of generated samples per kept sample).
"""
import torch


@torch.no_grad()
def density_ratio(D, x, eps=1e-6):
    """r(x) = p(x)/p_hat(x).

    With the BCE loss, the optimal discriminator is D*(x) = p/(p + p_hat),
    so p/p_hat = D/(1 - D). Clamping avoids division by zero when D saturates.
    """
    d = D(x).squeeze(1).clamp(eps, 1 - eps)
    return d / (1 - d)


def find_c(r, K, tol=1e-4, n_iter=100):
    """Bisection on log(c) so that mean(min(c*r, 1)) = 1/K."""
    if K <= 1:
        return float('inf')            # budget 1: accept everything
    target = 1.0 / K
    lo, hi = -30.0, 30.0               # search range for log(c)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        rate = torch.clamp(r * torch.exp(torch.tensor(mid)), max=1.0).mean().item()
        if abs(rate - target) < tol:
            break
        if rate > target:              # accepts too much -> decrease c
            hi = mid
        else:
            lo = mid
    return float(torch.exp(torch.tensor(mid)))


def find_threshold(r, K):
    """Ablation: keep only the top 1/K fraction of samples by r (deterministic rule, same budget)."""
    if K <= 1:
        return 0.0
    return torch.quantile(r.float(), 1.0 - 1.0 / K).item()


@torch.no_grad()
def calibrate(G, D, K, device, n=50_000, batch_size=2048, latent_dim=100):
    """Estimate c_K on n fresh samples from G. Returns (c_K, ratios)."""
    rs = []
    for i in range(0, n, batch_size):
        z = torch.randn(min(batch_size, n - i), latent_dim, device=device)
        rs.append(density_ratio(D, G(z)).cpu())
    r = torch.cat(rs)
    return find_c(r, K), r
