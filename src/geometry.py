"""
geometry.py -- flatness of the decision boundaries of trained conic heads (Section 7.8).

    python3 src/geometry.py "runs/CTRL_conic_bn_lr001_s*/final.pt" "runs/S_conic_nobn_s*/final.pt"

Class k owns the region where g_k - rho_k is smallest. By Proposition 6 the boundary
between two cells is a hyperplane only in a degenerate configuration, and a quadric
or quartic otherwise. For each class pair, the boundary is sampled within random
two-dimensional planes through the apex of one class, and the ratio of the smaller
to the larger singular value of the sampled points (about their mean) is recorded.
The ratio vanishes when the sampled curve is a straight line. The column
"curvature" is its mean over all pairs and planes.

Also printed: the coefficient of variation of the apertures (aCV), the mean angle
between the axes (tilt), and two scale-free ratios, eps/sep and rho/sep, where sep is
the mean distance between apices.
"""
from __future__ import annotations
import sys, glob, math
import torch
import torch.nn.functional as F


def head_params(ck):
    h = ck['head']
    if 'eta' not in h:
        return None
    a, w, eta = h['a'].double(), h['w'].double(), h['eta'].double()
    floor = float(ck.get('args', {}).get('tau_floor', 0.0) or 0.0)
    tau = floor + F.softplus(eta)
    return a, w, tau, tau * w.norm(dim=1), h['rho'].double(), \
        float(ck.get('args', {}).get('eps', 0.1))


def g(z, a, w, al, rho, eps):
    u = z - a
    return u @ w + al * torch.sqrt((u * u).sum(-1) + eps ** 2) - rho


def slice_boundary(k, l, a, w, al, rho, eps, e1, e2, n=400):
    """Sample the curve {g_k = g_l} inside the 2-plane through a[k] spanned by
    e1, e2.

    Along a ray z = a_k + t*dir the difference of the two cone functions is

        f(t) = A t + alpha_k sqrt(t^2 + eps^2)
                   - alpha_l sqrt(t^2 + 2 B t + C + eps^2) + D,

        A = <w_k - w_l, dir>,  B = <a_k - a_l, dir>,
        C = ||a_k - a_l||^2,   D = rho_l - rho_k - <w_l, a_k - a_l>,

    so once A, B, C, D are formed the bisection is pure scalar arithmetic and
    never touches a d-dimensional vector again. At d = 512 that is the whole
    cost of the measurement.
    """
    th = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
    D2 = torch.cos(th).unsqueeze(1) * e1 + torch.sin(th).unsqueeze(1) * e2   # (n,d)
    delta = a[k] - a[l]
    A = D2 @ (w[k] - w[l])
    B = D2 @ delta
    C = delta @ delta
    D = rho[l] - rho[k] - w[l] @ delta
    e2s = eps ** 2

    def f(t):
        return (A * t + al[k] * torch.sqrt(t * t + e2s)
                - al[l] * torch.sqrt(t * t + 2 * B * t + C + e2s) + D)

    lo = torch.full((n,), 1e-9, dtype=torch.float64)
    hi = torch.full((n,), 500.0, dtype=torch.float64)
    keep = (f(lo) * f(hi)) < 0
    if int(keep.sum()) < 40:
        return None
    lo, hi, A, B, th = lo[keep], hi[keep], A[keep], B[keep], th[keep]
    for _ in range(60):                       # 60 halvings of [1e-9, 500]
        m = 0.5 * (lo + hi)                   # resolves t to ~4e-16
        sgn = f(lo) * f(m) <= 0
        hi = torch.where(sgn, m, hi)
        lo = torch.where(sgn, lo, m)
    t = 0.5 * (lo + hi)
    return torch.stack([t * torch.cos(th), t * torch.sin(th)], 1)


def line_flatness(P):
    """rms distance from a planar point set to its best-fit line, divided by the
    rms spread of the set. 0 = the curve is straight."""
    X = P - P.mean(0)
    _, S, V = torch.linalg.svd(X, full_matrices=False)
    return float(S[1] / S[0])


def curvature(a, w, al, rho, eps, slices=6, seed=0):
    K, d = a.shape
    gen = torch.Generator().manual_seed(seed)
    vals = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            e1 = a[j] - a[i]
            if e1.norm() < 1e-12:
                continue
            e1 = e1 / e1.norm()
            for _ in range(slices):
                v = torch.randn(d, generator=gen, dtype=torch.float64)
                v = v - (v @ e1) * e1
                v = v / v.norm()
                P = slice_boundary(i, j, a, w, al, rho, eps, e1, v)
                if P is not None:
                    vals.append(line_flatness(P))
    return sum(vals) / len(vals) if vals else float('nan')


paths = [p for arg in sys.argv[1:] for p in sorted(glob.glob(arg))]
if not paths:
    raise SystemExit('usage: python3 src/geometry.py "runs/<run>/final.pt" [...]')

print('%-30s %6s %8s %8s %8s %9s %8s %8s' % (
    'checkpoint', 'BN', 'tau', 'aCV', 'tilt', 'curvature', 'eps/sep', 'rho/sep'))
for p in paths:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    hp = head_params(ck)
    if hp is None:
        print('%-30s  (not a conic head, skipped)' % p)
        continue
    a, w, tau, al, rho, eps = hp
    K = a.shape[0]
    cv = float(al.std() / al.mean())
    wn = F.normalize(w, dim=1)
    cos = (wn @ wn.t()).clamp(-1, 1)
    iu = torch.triu_indices(K, K, 1)
    ang = float(torch.rad2deg(torch.acos(cos[iu[0], iu[1]])).mean())
    cur = curvature(a, w, al, rho, eps)
    # Dimensionless groups that set the shape of the diagram: the boundary is
    # invariant when z, a and rho are scaled together, so these two ratios say how
    # large the apex smoothing and the offset spread are relative to the apices.
    iu2 = torch.triu_indices(K, K, 1)
    sep = float(torch.cdist(a, a)[iu2[0], iu2[1]].mean())
    eps_sep = eps / sep
    rho_sep = float(rho.std() / (al.mean() * sep)) if sep > 0 else float('nan')
    bn = 'no' if ck.get('args', {}).get('no_bn') else 'yes'
    name = p.split('/')[-2] if '/' in p else p
    print('%-30s %6s %8.3f %8.4f %7.1fd %9.4f %8.4f %8.4f' % (
        name, bn, float(tau.mean()), cv, ang, cur, eps_sep, rho_sep))
print('\ncurvature: 0 means the sampled boundary is a straight line (a hyperplane).')
