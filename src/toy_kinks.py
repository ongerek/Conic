"""
toy_kinks.py -- Proposition 11 on a small rectifier network, computed exactly (Section 6).

    python3 src/toy_kinks.py              about 30 seconds on one CPU core, no data needed

The network has 16 inputs, two hidden layers of 64 rectified units and 10 classes;
the data are ten Gaussian clusters. Along the line theta0 + t*delta, with delta drawn
filter-normalised, every pre-activation is a polynomial in t, so every breakpoint of
every sample is the root of a known polynomial (linear in the first layer, quadratic
in the second). For n = 16, 128 and 1024 samples the script prints the number of
kinks per unit length of t, the largest and the median kink relative to the mean
slope of the loss, and the largest relative error between the kink predicted by
eq. (kink) and the jump of one-sided derivatives of the black-box loss at the twelve
best-isolated kinks.
"""
import numpy as np

rng = np.random.default_rng(0)
m, H, K = 16, 64, 10
A1 = rng.standard_normal((H, m)) * np.sqrt(2 / m); c1 = 0.1 * rng.standard_normal(H)
A2 = rng.standard_normal((H, H)) * np.sqrt(2 / H); c2 = 0.1 * rng.standard_normal(H)
A3 = rng.standard_normal((K, H)) * np.sqrt(1 / H); c3 = np.zeros(K)


def fnorm(A, D):
    """Filter normalisation: each row of D rescaled to the norm of the row of A."""
    return D * np.linalg.norm(A, axis=1, keepdims=True) / np.linalg.norm(D, axis=1, keepdims=True)


D1, D2, D3 = [fnorm(A, rng.standard_normal(A.shape)) for A in (A1, A2, A3)]
mu = 3 * rng.standard_normal((K, m))


def data(n, seed):
    r = np.random.default_rng(seed); y = r.integers(0, K, n)
    return mu[y] + r.standard_normal((n, m)), y


def loss(t, X, y):
    """Black-box forward pass, with no knowledge of the breakpoints."""
    h1 = np.maximum(X @ (A1 + t * D1).T + c1, 0)
    h2 = np.maximum(h1 @ (A2 + t * D2).T + c2, 0)
    F = h2 @ (A3 + t * D3).T + c3
    F -= F.max(1, keepdims=True)
    return np.mean(np.log(np.exp(F).sum(1)) - F[np.arange(len(y)), y])


def F_dF(t, x, m1, m2):
    """Output and its t-derivative for fixed activation masks."""
    P1 = A1 @ x + c1; Q1 = D1 @ x; h1 = m1 * (P1 + t * Q1); dh1 = m1 * Q1
    p2 = (A2 + t * D2) @ h1 + c2; dp2 = D2 @ h1 + (A2 + t * D2) @ dh1
    h2 = m2 * p2; dh2 = m2 * dp2
    return (A3 + t * D3) @ h2 + c3, D3 @ h2 + (A3 + t * D3) @ dh2


def masks(t, x):
    p1 = (A1 + t * D1) @ x + c1; h1 = np.maximum(p1, 0); p2 = (A2 + t * D2) @ h1 + c2
    return (p1 > 0).astype(float), (p2 > 0).astype(float)


def breakpoints(x, lo, hi):
    """All t in (lo, hi) at which a pre-activation of sample x changes sign."""
    P1 = A1 @ x + c1; Q1 = D1 @ x
    r1 = -P1 / Q1; r1 = np.sort(r1[(r1 > lo) & (r1 < hi)]); out = list(r1)
    edges = np.r_[lo, r1, hi]
    for a, b in zip(edges[:-1], edges[1:]):
        m1 = ((P1 + 0.5 * (a + b) * Q1) > 0).astype(float)
        u = m1 * P1; v = m1 * Q1                        # h1 = u + t v on (a, b)
        c0 = A2 @ u + c2; cA = A2 @ v + D2 @ u; cB = D2 @ v   # p2 = c0 + cA t + cB t^2
        for j in range(H):
            for rt in np.roots([cB[j], cA[j], c0[j]]):
                if abs(rt.imag) < 1e-12 and a < rt.real < b:
                    out.append(rt.real)
    return np.sort(np.array(out))


def main():
    lo, hi = 0.5, 1.5
    print('%-6s %14s %26s %26s %24s' % ('n', 'kinks/unit t', 'largest kink/mean|slope|',
                                         'median kink/mean|slope|', 'eq. (kink) max rel. err'))
    for n in [16, 128, 1024]:
        X, y = data(n, n)
        allbp, jumps = [], []
        for i in range(n):
            bp = breakpoints(X[i], lo, hi); e = np.r_[lo, bp, hi]
            for k, tb in enumerate(bp):
                mL = masks(0.5 * (e[k] + tb), X[i]); mR = masks(0.5 * (tb + e[k + 2]), X[i])
                FL, dL = F_dF(tb, X[i], *mL); FR, dR = F_dF(tb, X[i], *mR)
                p = np.exp(FL - FL.max()); p /= p.sum(); g = p.copy(); g[y[i]] -= 1
                allbp.append(tb); jumps.append(g @ (dR - dL) / n)        # eq. (kink)
        allbp = np.array(allbp); jumps = np.array(jumps)
        o = np.argsort(allbp); allbp, jumps = allbp[o], jumps[o]
        ts = np.linspace(lo, hi, 4001)
        f = np.array([loss(t, X, y) for t in ts])
        slope = np.mean(np.abs(np.gradient(f, ts)))
        # independent check: one-sided derivatives of the black-box loss at the
        # best-isolated kinks (third-order one-sided differences)
        gaps = np.minimum(np.diff(np.r_[lo, allbp]), np.diff(np.r_[allbp, hi])); errs = []
        for k in np.argsort(-gaps)[:12]:
            tb, gp = allbp[k], gaps[k]; s = gp / 8
            fl = [loss(tb - j * s, X, y) for j in (0, 1, 2, 3)]
            fr = [loss(tb + j * s, X, y) for j in (0, 1, 2, 3)]
            dl = (11 * fl[0] - 18 * fl[1] + 9 * fl[2] - 2 * fl[3]) / (6 * s)
            dr = (-11 * fr[0] + 18 * fr[1] - 9 * fr[2] + 2 * fr[3]) / (6 * s)
            errs.append(abs((dr - dl) - jumps[k]) / max(abs(jumps[k]), 1e-12))
        print('%-6d %14.0f %26.2e %26.2e %24.1e' % (
            n, len(allbp) / (hi - lo), np.abs(jumps).max() / slope,
            np.median(np.abs(jumps)) / slope, max(errs)), flush=True)


if __name__ == '__main__':
    main()
