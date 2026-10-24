"""
kink_test.py -- crease test on refined windows of a parameter slice (Table 1).

With grid spacing h the normalised second difference

    D(h) = max |L[i+1] - 2 L[i] + L[i-1]| / h^2

tends to the finite curvature L'' on a smooth surface, but grows like dS/h across a
crease with slope jump dS. Fitting D ~ h^(-gamma) over the spacings h, 2h, 4h gives
gamma near 0 for a smooth surface and near 1 for a crease. The slope jump is
estimated as h * D(h) at the finest spacing and compared with the typical slope.
The verdict uses the finest pair of spacings: D(h)/D(2h) is about 1 for a smooth
surface, 2 for a crease, and 4 for a feature narrower than the grid, in which case
no verdict is given.

    python3 src/kink_test.py results/slices/R56_*_zoom_*.csv
"""
import sys
import numpy as np


def load(fn):
    d = np.genfromtxt(fn, delimiter=',', names=True)
    xs = np.unique(d['x']); ys = np.unique(d['y'])
    return xs, ys, d['z'].reshape(len(xs), len(ys))


def scaling(Z, h0, strides=(1, 2, 4), core=0.5):
    """D(h) over the central part of the window, along both grid axes."""
    g = Z.shape[0]
    lo, hi = int(g * (1 - core) / 2), int(g * (1 + core) / 2)
    out = []
    for s in strides:
        S = Z[::s, ::s]; h = h0 * s
        a, b = lo // s, max(hi // s, lo // s + 3)
        d2x = np.abs(S[2:, 1:-1] - 2 * S[1:-1, 1:-1] + S[:-2, 1:-1])[a:b, a:b]
        d2y = np.abs(S[1:-1, 2:] - 2 * S[1:-1, 1:-1] + S[1:-1, :-2])[a:b, a:b]
        out.append((h, max(d2x.max(), d2y.max()) / h ** 2))
    hs, Ds = np.array(out).T
    p = -np.polyfit(np.log(hs), np.log(Ds), 1)[0]
    return hs, Ds, p


def report(name, xs, ys, Z):
    """Verdict from the FINEST pair of spacings, where the surface is best
    resolved. With r = D(h)/D(2h):  r ~ 1 smooth,  r ~ 2 crease,  r ~ 4 a
    feature narrower than the grid. A single exponent fitted over all spacings
    lets an unresolved coarse step masquerade as a crease, so it is shown for
    information only."""
    h0 = xs[1] - xs[0]
    hs, Ds, p_fit = scaling(Z, h0)
    p = np.log2(Ds[0] / Ds[1])                 # local exponent at the finest step
    gx, gy = np.gradient(Z, xs, ys)
    slope = np.median(np.hypot(gx, gy))
    jump = hs[0] * Ds[0]                       # slope jump, or the detection floor
    print('%-26s D(h) at h=%s : %s' % (name, ' '.join('%.4f' % h for h in hs),
                                       ' '.join('%.3g' % D for D in Ds)))
    head = '%26s finest-step exponent p = %.2f (fit over all spacings %.2f) -> ' % ('', p, p_fit)
    if p > 1.5:
        print(head + 'UNDECIDED: a feature narrower than the grid; refine the grid')
    elif p > 0.5:
        print(head + 'crease (cone-like) | slope jump ~ %.3g = %.0f%% of the typical slope %.3g'
              % (jump, 100 * jump / slope, slope))
    elif jump / slope > 0.3:
        print(head + 'UNDECIDED: grid too coarse, a crease up to %.0f%% of the typical '
              'slope %.3g could hide here' % (100 * jump / slope, slope))
    else:
        print(head + 'smooth ripple | no crease larger than ~%.3g (%.0f%% of the typical slope %.3g)'
              % (jump, 100 * jump / slope, slope))


if __name__ == '__main__':
    import os
    files = [f for f in sys.argv[1:] if os.path.exists(f)]
    if not files:
        raise SystemExit('no zoom CSVs found; run the zoom step of scripts/analyze.sh first')
    for fn in files:
        report(fn.split('/')[-1], *load(fn))
