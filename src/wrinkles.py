"""
wrinkles.py -- do the wrinkles of a backbone slice depend on the number of images?
(Section 6: 512 against 4096 training images.)

    python3 src/wrinkles.py results/slices/R56_skip_param_backbone_train.csv \
                            results/slices/R56_skip_param_backbone_train_4096.csv

The first file is the 41 x 41 slice over 512 images, the second a 21 x 21 slice of
the same plane over 4096 images; every second point of the first grid coincides with
a point of the second. The wrinkles are the residual of each surface after a Gaussian
smoothing of width one grid step. If they came from the sample of images, their
amplitude would fall by about sqrt(512/4096) = 0.35; if they belong to the network,
their amplitude and position stay. Needs numpy and scipy.
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter


def load(fn):
    d = np.genfromtxt(fn, delimiter=',', names=True)
    xs = np.unique(d['x']); g = len(xs)
    return xs, d['z'].reshape(g, g)


def wrinkle(Z, sig=1.0):
    R = (Z - gaussian_filter(Z, sigma=sig, mode='nearest'))[2:-2, 2:-2]
    return np.sqrt(np.mean(R ** 2)), R


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    xa, Za = load(sys.argv[1]); xb, Zb = load(sys.argv[2])
    step = (len(xa) - 1) // (len(xb) - 1)
    Za = Za[::step, ::step]
    if not np.allclose(xa[::step], xb):
        raise SystemExit('the two grids do not share their points')
    if Zb.shape[0] < 7:
        raise SystemExit('the grids are too coarse (the second needs at least 7 x 7 points)')
    wa, Ra = wrinkle(Za); wb, Rb = wrinkle(Zb)
    print('surfaces on %d common grid points: correlation %.4f' % (Za.size, np.corrcoef(Za.ravel(), Zb.ravel())[0, 1]))
    print('wrinkle rms: %.4f (first file) and %.4f (second file), ratio %.2f'
          % (wa, wb, wb / wa))
    print('correlation of the two wrinkle maps: %.3f' % np.corrcoef(Ra.ravel(), Rb.ravel())[0, 1])


if __name__ == '__main__':
    main()
