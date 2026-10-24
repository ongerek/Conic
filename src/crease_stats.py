"""
crease_stats.py -- the Section 6 numbers of the twenty planes and forty lines, from
the output of crease_lines.py.

    python3 src/crease_stats.py results/crease

Prints, for each network, whether the trained point is the minimum of every coarse
slice and how many deep minima each slice has, the range of the loss at radius 0.9;
over all lines, the exponent gamma (median, bootstrap 95% interval of the median,
range), the kinks per unit length and per grid step; and the bound 4 h D(h) on any
crease, relative to the local slope on the walls and relative to the wall slope of
the same plane on the basin floor.
"""
import argparse, csv, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', nargs='?', default='results/crease')
    ap.add_argument('--boot', type=int, default=10000)
    a = ap.parse_args()
    P = list(csv.DictReader(open(os.path.join(a.folder, 'crease_planes.csv'))))
    L = list(csv.DictReader(open(os.path.join(a.folder, 'crease_lines.csv'))))
    f = lambda r, k: float(r[k])
    nets = list(dict.fromkeys(r['run'] for r in P))
    for net in nets:
        pp = [r for r in P if r['run'] == net]
        at0 = all(f(r, 'argmin_x') == 0 and f(r, 'argmin_y') == 0 for r in pp)
        dm = sorted(set(int(r['deep_minima']) for r in pp))
        print('%-12s %2d planes | trained point is the slice minimum in all: %s | deep minima per '
              'slice: %s | loss at radius 0.9: %.1f to %.1f'
              % (net, len(pp), 'yes' if at0 else 'NO', dm,
                 min(f(r, 'ray09_min') for r in pp), max(f(r, 'ray09_max') for r in pp)))
    g = np.array([f(r, 'gamma_fit') for r in L])
    rng = np.random.default_rng(0)
    bs = np.median(rng.choice(g, (a.boot, len(g))), axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('gamma over %d lines: median %.2f, bootstrap 95%% interval %.2f to %.2f, range %.2f to %.2f'
          % (len(g), np.median(g), lo, hi, g.min(), g.max()))
    kpu = np.array([f(r, 'kinks_per_unit') for r in L])
    # kinks between neighbouring points = kinks per unit length times the spacing h
    steps = np.array([f(r, 'kinks_per_unit') * f(r, 'h') for r in L])
    print('kinks per unit length: median %.2g (range %.2g to %.2g); per grid step: median %.2g'
          % (np.median(kpu), kpu.min(), kpu.max(), np.median(steps)))
    for net in nets:
        w = [r for r in L if r['run'] == net and r['line'] == 'wall']
        fl_ = [r for r in L if r['run'] == net and r['line'] == 'floor']
        b = np.array([4 * f(r, 'floor_over_slope') for r in w])
        rat = []
        for r in fl_:
            wr = [x for x in w if x['plane'] == r['plane']]
            if wr:
                rat.append(4 * f(r, 'h') * f(r, 'D_h') / f(wr[0], 'slope'))
        print('%-12s crease bound 4hD(h) on the walls: median %.1f%%, largest %.1f%% of the local slope;'
              ' on the floor at most %.2g of the wall slope'
              % (net, 100 * np.median(b), 100 * b.max(), max(rat) if rat else float('nan')))


if __name__ == '__main__':
    main()
