"""
crease_lines.py -- planes and refined lines of the training loss through a trained
backbone (Section 6: twenty planes, forty lines, kink counts).

    python3 src/crease_lines.py --selftest                       about a minute
    python3 src/crease_lines.py "runs/R56_skip/final.pt" "runs/R56_noskip/final.pt"

For each checkpoint and each of --planes random planes through the trained point
(two filter-normalised directions over all parameters, the final layer included;
plane p draws its directions with generator seed p), it measures

  1. a coarse slice of the training loss on a --coarse x --coarse grid over
     [-span, span]^2: the number of deep local minima (interior grid points lower
     than their eight neighbours, with loss below log K), the position of the slice
     minimum, and the lowest and highest loss on 72 rays at radius 0.9;
  2. two refined lines along the first direction of the plane, one through the
     trained point (the basin floor) and one centred at radius --wall (a wall),
     each with --points points over a half-width --halfwidth. On each line it
     reports the exponent gamma of D(h) = max|L[i+1]-2L[i]+L[i-1]|/h^2 over the
     spacings h, 2h, 4h, the exponent from the finest pair, the median slope, and
     h*D(h) relative to that slope (the bound on a crease is 4 h D(h));
  3. on each refined line, the number of activation-pattern changes between
     neighbouring points, summed over every rectifier and every image. Each change
     is one kink of Proposition 11, carrying the weight of one image.

The loss is averaged over the first --images training images without augmentation,
with batch normalization in inference mode. Lines are computed in float64 with
deterministic cuDNN, so that rounding does not enter the second differences.

Writes <out>/crease_planes.csv and <out>/crease_lines.csv (default results/crease).
"""
from __future__ import annotations
import argparse, csv, glob, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import make_backbone, make_head, CIFAR_MEAN, CIFAR_STD

# ---------------------------------------------------------------- relu recorder
_relu = Fn.relu
_record = None                      # list that collects (input > 0) while active


def _relu_recording(input, inplace=False):
    if _record is not None:
        _record.append(input > 0)
    return _relu(input, inplace=inplace)


Fn.relu = _relu_recording           # nn.ReLU and the F.relu calls of the blocks use it


# ---------------------------------------------------------------- model and data
def rebuild(ck, dev):
    a = ck['args']
    K = 10 if a.get('dataset', 'cifar10') == 'cifar10' else 100
    net = make_backbone(a.get('backbone', 'resnet18'), not a.get('no_bn', False),
                        not a.get('no_skip', False))
    head = make_head(a['head'], net.feat_dim, K, a.get('eps', 0.1), a.get('tau_init', 1.5),
                     not a.get('fix_tau', False), a.get('tau_floor', 0.0))
    net.load_state_dict(ck['net']); head.load_state_dict(ck['head'])
    return net.to(dev).eval(), head.to(dev).eval(), K


def train_images(dataset, n, data_dir, fake):
    if fake:
        g = torch.Generator().manual_seed(0)
        return torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    cls = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}[dataset]
    ds = cls(data_dir, train=True, download=True, transform=tf)
    xs, ys = zip(*[ds[i] for i in range(n)])
    return torch.stack(xs), torch.tensor(ys)


def filter_normalised_direction(params, gen):
    """As in loss_surface.py: one random tensor per parameter tensor, each filter
    (row) rescaled to the norm of the trained filter; biases and norm parameters
    are held fixed."""
    d = []
    for p in params:
        if p.dim() <= 1:
            d.append(torch.zeros_like(p)); continue
        r = torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.device, p.dtype)
        rf, pf = r.reshape(p.shape[0], -1), p.reshape(p.shape[0], -1)
        rf.mul_(pf.norm(dim=1, keepdim=True) / (rf.norm(dim=1, keepdim=True) + 1e-12))
        d.append(rf.reshape(p.shape))
    return d


class Slice:
    def __init__(self, net, head, X, Y, chunk):
        self.net, self.head, self.X, self.Y, self.chunk = net, head, X, Y, chunk
        self.params = list(net.parameters()) + list(head.parameters())
        self.theta0 = [p.detach().clone() for p in self.params]

    def to(self, dtype):
        self.net.to(dtype); self.head.to(dtype)
        self.params = list(self.net.parameters()) + list(self.head.parameters())
        self.theta0 = [t.to(dtype) for t in self.theta0]
        self.X = self.X.to(dtype)
        return self

    @torch.no_grad()
    def loss(self, d1, d2, x, y, record=False):
        global _record
        for p, t0, a, b in zip(self.params, self.theta0, d1, d2):
            p.copy_(t0 + x * a.to(t0.dtype) + y * b.to(t0.dtype))
        _record = [] if record else None
        tot = 0.0
        for i in range(0, self.X.shape[0], self.chunk):
            out = self.head(self.net(self.X[i:i + self.chunk]))
            tot += float(Fn.cross_entropy(out, self.Y[i:i + self.chunk], reduction='sum'))
        pattern, _record = _record, None
        return tot / self.X.shape[0], pattern


def flips(a, b):
    return sum(int((u ^ v).sum()) for u, v in zip(a, b))


def gamma_1d(L, h0, strides=(1, 2, 4), core=0.5):
    P = len(L); lo, hi = int(P * (1 - core) / 2), int(P * (1 + core) / 2)
    hs, Ds = [], []
    for s in strides:
        S = L[::s]; a, b = lo // s, max(hi // s, lo // s + 3)
        d2 = np.abs(S[2:] - 2 * S[1:-1] + S[:-2])[a:b]
        hs.append(h0 * s); Ds.append(d2.max() / (h0 * s) ** 2)
    hs, Ds = np.array(hs), np.array(Ds)
    g_fit = -np.polyfit(np.log(hs), np.log(Ds), 1)[0]
    g_fine = np.log2(Ds[0] / Ds[1])
    slope = float(np.median(np.abs(np.gradient(L, h0)[lo:hi])))
    return g_fit, g_fine, Ds, slope, hs[0] * Ds[0]


# --------------------------------------------------------------------- main
def run(args, dev):
    paths = []
    for pat in args.ckpts:
        hit = sorted(glob.glob(pat))
        if not hit:
            print(f'  note: nothing matches {pat} -- skipped')
        paths += hit
    if not paths:
        raise SystemExit('no checkpoints matched')
    os.makedirs(args.out, exist_ok=True)
    plane_rows, line_rows = [], []
    for path in paths:
        name = os.path.basename(os.path.dirname(path)) or path
        ck = torch.load(path, map_location='cpu', weights_only=False)
        net, head, K = rebuild(ck, dev)
        X, Y = train_images(ck['args'].get('dataset', 'cifar10'), args.images, args.data_dir,
                            args.fake_data)
        sl = Slice(net, head, X.to(dev), Y.to(dev), args.chunk)
        for plane in range(args.planes):
            t0 = time.time()
            sl.to(torch.float32)
            gen = torch.Generator().manual_seed(plane)
            d1 = filter_normalised_direction(sl.theta0, gen)
            d2 = filter_normalised_direction(sl.theta0, gen)
            # 1. coarse slice and rays, float32
            ts = np.linspace(-args.span, args.span, args.coarse)
            G = np.array([[sl.loss(d1, d2, float(x), float(y))[0] for y in ts] for x in ts])
            mins = 0
            for i in range(1, args.coarse - 1):
                for j in range(1, args.coarse - 1):
                    nb = G[i - 1:i + 2, j - 1:j + 2].copy(); nb[1, 1] = np.inf
                    mins += int(G[i, j] < nb.min() and G[i, j] < math.log(K))
            imin = np.unravel_index(G.argmin(), G.shape)
            ray = [sl.loss(d1, d2, 0.9 * math.cos(a), 0.9 * math.sin(a))[0]
                   for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)]
            plane_rows.append(dict(run=name, plane=plane, loss_at_trained=sl.loss(d1, d2, 0.0, 0.0)[0],
                                   slice_min=float(G.min()), argmin_x=float(ts[imin[0]]),
                                   argmin_y=float(ts[imin[1]]), deep_minima=mins,
                                   ray09_min=min(ray), ray09_max=max(ray)))
            # 2.-3. refined lines, float64
            sl.to(torch.float64)
            d1 = [t.double() for t in d1]; d2 = [t.double() for t in d2]
            for kind, c in (('floor', 0.0), ('wall', args.wall)):
                tt = np.linspace(c - args.halfwidth, c + args.halfwidth, args.points)
                L, nflip, prev = [], [], None
                for t in tt:
                    v, pat = sl.loss(d1, d2, float(t), 0.0, record=True)
                    L.append(v)
                    if prev is not None:
                        nflip.append(flips(prev, pat))
                    prev = pat
                L = np.array(L); h0 = tt[1] - tt[0]
                g_fit, g_fine, Ds, slope, jump = gamma_1d(L, h0)
                line_rows.append(dict(run=name, plane=plane, line=kind, centre=c, h=h0,
                                      loss_mean=float(L.mean()), gamma_fit=g_fit, gamma_finest=g_fine,
                                      D_h=Ds[0], D_2h=Ds[1], D_4h=Ds[2], slope=slope,
                                      floor_over_slope=jump / slope if slope > 0 else float('nan'),
                                      kinks=int(sum(nflip)), kinks_per_unit=sum(nflip) / (tt[-1] - tt[0]),
                                      max_kinks_per_step=int(max(nflip)), images=args.images))
            sl.to(torch.float32)
            last = line_rows[-2:]
            print(f'  {name} plane {plane}: deep minima {mins}, floor gamma {last[0]["gamma_fit"]:.2f}, '
                  f'wall gamma {last[1]["gamma_fit"]:.2f}, kinks/unit {last[1]["kinks_per_unit"]:.3g} '
                  f'({time.time() - t0:.0f} s)', flush=True)
        for p, t0 in zip(sl.params, sl.theta0):     # leave the model as it was loaded
            p.data.copy_(t0)

    def dump(rows, fname):
        cols = list(rows[0])
        with open(os.path.join(args.out, fname), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows:
                w.writerow({c: (f'{v:.6g}' if isinstance(v, float) else v) for c, v in r.items()})
        print(f'wrote {args.out}/{fname} ({len(rows)} rows)')
    dump(plane_rows, 'crease_planes.csv'); dump(line_rows, 'crease_lines.csv')
    print('CREASE DONE.')
    return plane_rows, line_rows


def selftest(dev):
    """A random ResNet-56 checkpoint on random images, with tiny settings."""
    import tempfile, types
    ok = True
    # gamma estimator on known curves: a kink gives about 1, a parabola about 0
    t = np.linspace(-1, 1, 257); h = t[1] - t[0]
    g_kink = gamma_1d(np.abs(t - 0.0013) + 0.1 * t ** 2, h)[0]
    g_smooth = gamma_1d(t ** 2 + 0.01 * np.sin(5 * t), h)[0]
    print(f'  gamma of a kink {g_kink:.2f} (about 1), of a smooth curve {g_smooth:.2f} (about 0)')
    ok &= 0.8 < g_kink < 1.2 and abs(g_smooth) < 0.2
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0)
        a = dict(dataset='cifar10', backbone='resnet56', head='affine', no_bn=False,
                 no_skip=False, eps=0.1, tau_init=1.5, fix_tau=False, tau_floor=0.0)
        net = make_backbone('resnet56', True, True)
        head = make_head('affine', net.feat_dim, 10, 0.1, 1.5, True, 0.0)
        os.makedirs(os.path.join(tmp, 'runs', 'ST'))
        torch.save({'net': net.state_dict(), 'head': head.state_dict(), 'args': a},
                   os.path.join(tmp, 'runs', 'ST', 'final.pt'))
        args = types.SimpleNamespace(ckpts=[os.path.join(tmp, 'runs', 'ST', 'final.pt')],
                                     planes=1, images=16, chunk=16, coarse=5, span=1.0,
                                     points=17, halfwidth=0.02, wall=0.5, out=os.path.join(tmp, 'o'),
                                     data_dir='./data', fake_data=True)
        planes, lines = run(args, dev)
        ok &= len(planes) == 1 and len(lines) == 2 and lines[0]['kinks'] >= 0
    print('SELFTEST PASSED' if ok else 'SELFTEST FAILED')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--planes', type=int, default=10)
    ap.add_argument('--images', type=int, default=512)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--coarse', type=int, default=21)
    ap.add_argument('--span', type=float, default=1.0)
    ap.add_argument('--points', type=int, default=257)
    ap.add_argument('--halfwidth', type=float, default=0.02)
    ap.add_argument('--wall', type=float, default=0.5)
    ap.add_argument('--out', default='results/crease')
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--device', default=None)
    ap.add_argument('--fake-data', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    dev = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if dev.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print(f'device {dev}')
    if args.selftest:
        raise SystemExit(0 if selftest(dev) else 1)
    run(args, dev)


if __name__ == '__main__':
    main()
