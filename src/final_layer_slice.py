"""
final_layer_slice.py -- the Section 5 measurements on the two final-layer slices of
Fig. 2 (written by loss_surface.py --mode param --scope head).

    python3 src/final_layer_slice.py results/slices/S_affine_bn_s0_param_head_span40.csv \
                                     results/slices/S_affine_bn_s0_param_head_span150.csv

For each slice: the position of the minimum, the loss at the trained point, and the
smallest second difference along the two axes and the two diagonals (convexity
requires it to be non-negative). On the wider slice (the last file given):
  - the chord slope (L(r) - L(0)) / r along 16 rays through grid points, without
    interpolation (Proposition 10: nondecreasing, levelling off at the wall slope);
  - along 72 rays, with cubic interpolation, the growth exponent p of L - L(0) ~ t^p
    in the bands [5,10], [10,20], [20,40], [40,80], [80,150];
  - a rounded cone sigma ((t^2 + r_c^2)^(1/2) - r_c) fitted along each ray: core
    radius r_c and wall slope sigma;
  - how far the loss stays below the straight wall L(0) + sigma t (the band of
    eq. (sandwich) must keep this bounded);
  - whether the level set at height one of the asymptotic cone, rebuilt from the 72
    limiting chord slopes, is convex.
Needs numpy and scipy.
"""
import argparse
import numpy as np


def load(f):
    M = np.genfromtxt(f, delimiter=',', names=True)
    x, y, z = M['x'], M['y'], M['z']
    xs, ys = np.unique(x), np.unique(y)
    Z = np.full((len(ys), len(xs)), np.nan)
    Z[np.searchsorted(ys, y), np.searchsorted(xs, x)] = z
    if np.isnan(Z).any():
        raise SystemExit('%s is not a full rectangular grid' % f)
    return xs, ys, Z


def second_differences(Z):
    return [Z[:, 2:] - 2 * Z[:, 1:-1] + Z[:, :-2], Z[2:, :] - 2 * Z[1:-1, :] + Z[:-2, :],
            Z[2:, 2:] - 2 * Z[1:-1, 1:-1] + Z[:-2, :-2], Z[2:, :-2] - 2 * Z[1:-1, 1:-1] + Z[:-2, 2:]]


def grid_rays(xs, Z):
    """Chord slopes along 16 rays through grid points of a square grid centred at 0."""
    c = len(xs) // 2; L0 = Z[c, c]; h = xs[1] - xs[0]; ends, incs = [], []
    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1),
                   (2, 1), (1, 2), (-2, 1), (-1, 2), (-2, -1), (-1, -2), (2, -1), (1, -2)]:
        n = c // max(abs(dx), abs(dy)); k = np.arange(1, n + 1)
        if n < 2:
            continue
        L = Z[c + dy * k, c + dx * k]; r = k * np.hypot(dx, dy) * h; cs = (L - L0) / r
        ends.append(cs[-1]); incs.append(np.diff(cs).min())
    if not incs:
        raise SystemExit('the slice is too coarse for the ray test (at least 5 x 5 points are needed)')
    return np.array(ends), min(incs)


def main():
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    ap = argparse.ArgumentParser()
    ap.add_argument('slices', nargs='+', help='one or more slice CSVs; the last is the wide one')
    a = ap.parse_args()
    for f in a.slices:
        xs, ys, Z = load(f)
        i = np.unravel_index(np.argmin(Z), Z.shape); c = len(xs) // 2
        d2 = min(float(v.min()) for v in second_differences(Z))
        ends, inc = grid_rays(xs, Z)
        print('%s\n  grid %dx%d, step %.3g | loss %.4g to %.4g | minimum at (%.3g, %.3g) | at the '
              'trained point %.4g\n  smallest second difference (axes and diagonals) %.2e | 16 grid rays: '
              'smallest increment of the chord slope %.2e, end slopes %.2f to %.2f (ratio %.1f)'
              % (f, len(xs), len(ys), xs[1] - xs[0], Z.min(), Z.max(), xs[i[1]], ys[i[0]], Z[c, c],
                 d2, inc, ends.min(), ends.max(), ends.max() / ends.min()))

    xs, ys, Z = load(a.slices[-1])
    I = RegularGridInterpolator((ys, xs), Z, method='cubic')
    i0, j0 = np.argmin(np.abs(ys)), np.argmin(np.abs(xs))
    L0 = float(Z[i0, j0]) if ys[i0] == 0 and xs[j0] == 0 else float(I([[0, 0]])[0])   # trained point
    R = min(abs(xs[0]), xs[-1], abs(ys[0]), ys[-1])
    rr = np.linspace(R / 300, R, 600); th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rays = [I(np.c_[rr * np.sin(t), rr * np.cos(t)]) - L0 for t in th]
    print('\n72 interpolated rays on the wide slice (radius up to %.3g):' % R)
    bands = [(b0, b1) for b0, b1 in [(5, 10), (10, 20), (20, 40), (40, 80), (80, 150)] if b1 <= R + 1e-9]
    for b0, b1 in bands:
        m = (rr >= b0) & (rr <= b1)
        ex = [np.polyfit(np.log(rr[m]), np.log(np.maximum(L[m], 1e-12)), 1)[0] for L in rays]
        print('  growth exponent in [%3g, %3g]: median %.2f (range %.2f to %.2f)'
              % (b0, b1, np.median(ex), min(ex), max(ex)))
    cores, slopes, below = [], [], []
    for L in rays:
        res = least_squares(lambda p: p[0] * (np.sqrt(rr ** 2 + p[1] ** 2) - abs(p[1])) - L, [0.5, 8.0])
        s, rc = res.x[0], abs(res.x[1]); cores.append(rc); slopes.append(s)
        below.append(L - s * rr)
    print('  rounded cone: core radius %.1f to %.1f (median %.1f), wall slope %.2f to %.2f'
          % (min(cores), max(cores), np.median(cores), min(slopes), max(slopes)))
    print('  loss rises from %.3g to %.3g; it stays below the straight wall L(0) + sigma t by %.3g to %.3g'
          % (L0, L0 + max(L.max() for L in rays), -max(b.max() for b in below), -min(b.min() for b in below)))
    s_inf = np.array([L[-1] / rr[-1] for L in rays])
    P = np.c_[np.cos(th) / s_inf, np.sin(th) / s_inf]
    hull = ConvexHull(P)
    on = all(np.min(np.abs(hull.equations[:, :2] @ p + hull.equations[:, 2])) < 1e-3 * np.abs(P).max()
             for p in P)
    print('  level set at height one of the asymptotic cone (72 limiting chord slopes): convex: %s'
          % ('yes' if on else 'no'))


if __name__ == '__main__':
    main()
