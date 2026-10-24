"""
loss_surface.py -- two-dimensional slices of the loss of a trained checkpoint.

  --mode param     the loss as a function of the PARAMETERS on a plane through the
                   trained point, spanned by two filter-normalised random directions
                   (Li et al., NeurIPS 2018). --scope chooses what moves: all,
                   backbone, head, or a substring of the parameter names.
                   Fig. 2 uses --scope head; Fig. 3 and Table 1 use --scope backbone.

  --mode feature   the winning score and the decision cells as functions of the
                   FEATURE on a plane through three class apices (or the top two
                   principal directions of the apices). The backbone runs once.

Examples (the commands of scripts/analyze.sh)
    python3 src/loss_surface.py runs/S_affine_bn_s0/final.pt --mode param --scope head \
            --span 40 --grid 41 --out results/slices/S_affine_bn_s0_param_head_span40.csv
    python3 src/loss_surface.py runs/R56_skip/final.pt --mode param --scope backbone \
            --split train --grid 41 --images 512

Writes a CSV with columns x,y,z (and cell in feature mode), which MATLAB reads with
readmatrix or dlmread, and a PNG preview if matplotlib is installed. Without --out
the CSV goes to results/slices/<run>_<mode>_<scope>[_train].csv.
Batch normalization runs in inference mode throughout.
"""
from __future__ import annotations
import argparse, copy, csv, math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import make_backbone, make_head, get_data, CIFAR_MEAN, CIFAR_STD


# ---------------------------------------------------------------- rebuilding
def rebuild(ck, dev):
    a = ck['args']
    K = 10 if a['dataset'] == 'cifar10' else 100
    net = make_backbone(a['backbone'], not a['no_bn'],
                        not a.get('no_skip', False)).to(dev)
    head = make_head(a['head'], net.feat_dim, K, a['eps'], a['tau_init'],
                     not a['fix_tau'], a.get('tau_floor', 0.0)).to(dev)
    net.load_state_dict(ck['net']); head.load_state_dict(ck['head'])
    net.eval(); head.eval()
    return net, head, a, K


# ------------------------------------------------------------- evaluation data
def eval_loader(cfg, args, dev):
    """Images on which the surface is evaluated.

    test : the held-out set, as in the earlier slices.
    train: the training set, which is the landscape the optimiser actually walks
           and the one Li et al. (2018) plot. It is read WITHOUT the random crops
           and flips used in training, in a fixed order, so every grid point sees
           exactly the same images."""
    if args.split == 'test' or args.fake_data:
        _, ld, _ = get_data(cfg['dataset'], args.data_dir, 256, 0, dev, args.fake_data)
        return ld
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    cls = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}[cfg['dataset']]
    ds = cls(args.data_dir, train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)


# ---------------------------------------------------------- parameter slice
def filter_normalised_direction(params, gen):
    """Li et al.: one random direction per parameter tensor, with each filter
    (row) rescaled to the norm of the corresponding filter of the trained
    weights, so that slices of differently scaled networks can be compared."""
    d = []
    for p in params:
        if p.dim() <= 1:                       # biases and norm parameters
            d.append(torch.zeros_like(p))      # are held fixed, as in the paper
            continue
        r = torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.device, p.dtype)
        rf = r.reshape(p.shape[0], -1)
        pf = p.reshape(p.shape[0], -1)
        rf.mul_(pf.norm(dim=1, keepdim=True) / (rf.norm(dim=1, keepdim=True) + 1e-12))
        d.append(rf.reshape(p.shape))
    return d


def select(net, head, scope):
    named = [('net.' + n, p) for n, p in net.named_parameters()] + \
            [('head.' + n, p) for n, p in head.named_parameters()]
    if scope == 'all':
        sel = named
    elif scope == 'backbone':
        sel = [x for x in named if x[0].startswith('net.')]
    elif scope == 'head':
        sel = [x for x in named if x[0].startswith('head.')]
    else:
        sel = [x for x in named if scope in x[0]]
    if not sel:
        raise SystemExit('no parameters matched --scope %r. available:\n  %s'
                         % (scope, '\n  '.join(n for n, _ in named)))
    return sel


@torch.no_grad()
def eval_loss(net, head, batches, dev):
    tot = n = 0.0
    for x, y in batches:
        out = head(net(x))
        tot += F.cross_entropy(out, y, reduction='sum').item()
        n += y.numel()
    return tot / n


@torch.no_grad()
def eval_head(head, feat_batches):
    tot = n = 0.0
    for z, y in feat_batches:
        tot += F.cross_entropy(head(z), y, reduction='sum').item()
        n += y.numel()
    return tot / n


def param_surface(ck, dev, args):
    net, head, cfg, K = rebuild(ck, dev)
    sel = select(net, head, args.scope)
    names = [n for n, _ in sel]
    print('moving %d tensors (%s), %.2fM parameters'
          % (len(sel), args.scope, sum(p.numel() for _, p in sel) / 1e6))

    test_ld = eval_loader(cfg, args, dev)
    batches, seen = [], 0
    for x, y in test_ld:
        batches.append((x.to(dev), y.to(dev))); seen += y.numel()
        if seen >= args.images:
            break
    print('evaluating on %d images per grid point' % seen)

    # If only head parameters move, the backbone output never changes, so the
    # backbone runs once and the head is evaluated on cached features.
    head_only = all(n.startswith('head.') for n in names)
    if head_only:
        with torch.no_grad():
            batches = [(net(x), y) for x, y in batches]
        print('head-only scope: backbone features cached')

    gen = torch.Generator().manual_seed(args.seed)      # CPU generator, portable
    params = [p for _, p in sel]
    theta0 = [p.detach().clone() for p in params]
    d1 = filter_normalised_direction(theta0, gen)
    d2 = filter_normalised_direction(theta0, gen)

    g = args.grid
    # --center moves the window along the SAME plane (same seed, same
    # directions), so a zoom lines up exactly with the full slice
    ca, cb = args.center
    ts = torch.linspace(-args.span, args.span, g, dtype=torch.float64)
    xs, ys = ca + ts, cb + ts
    Z = torch.zeros(g, g)
    for i, ai in enumerate(xs):
        for j, bj in enumerate(ys):
            with torch.no_grad():
                for p, p0, u, v in zip(params, theta0, d1, d2):
                    p.copy_(p0 + float(ai) * u + float(bj) * v)
            Z[i, j] = (eval_head(head, batches) if head_only
                       else eval_loss(net, head, batches, dev))
        print('  row %2d/%d  min %.4f  max %.4f' % (i + 1, g, Z[i].min(), Z[i].max()))
    with torch.no_grad():
        for p, p0 in zip(params, theta0):
            p.copy_(p0)
    return xs, ys, Z, None


# ------------------------------------------------------------ feature slice
def feature_surface(ck, dev, args):
    net, head, cfg, K = rebuild(ck, dev)

    # All plane geometry is done on the CPU in float64; only the backbone and the
    # head run on the device.
    if hasattr(head, 'a'):
        P = head.a.detach()
    elif hasattr(head, 'fc'):
        P = head.fc.weight.detach()
    else:
        P = head.w.detach()
    P = P.cpu().double()
    if args.plane == 'pca':
        c = P.mean(0)
        _, _, V = torch.linalg.svd(P - c, full_matrices=False)
        e1, e2 = V[0], V[1]                   # top two principal directions
    else:
        i, j, k = (int(v) for v in args.plane.split(','))
        c = (P[i] + P[j] + P[k]) / 3          # plane through three prototypes:
        e1 = P[j] - P[i]                      # guarantees those cells appear
        e1 = e1 / e1.norm()
        e2 = P[k] - c
        e2 = e2 - (e2 @ e1) * e1
        e2 = e2 / e2.norm()

    test_ld = eval_loader(cfg, args, dev)
    feats, labs, seen = [], [], 0
    with torch.no_grad():
        for x, y in test_ld:
            feats.append(net(x.to(dev)).cpu()); labs.append(y.cpu()); seen += y.numel()
            if seen >= args.images:
                break
    Zf = torch.cat(feats).double(); Yf = torch.cat(labs)
    p1 = (Zf - c) @ e1; p2 = (Zf - c) @ e2
    # the window covers the apices as well as the data, so that every cell of the
    # plane appears on the slice
    q1 = (P - c) @ e1; q2 = (P - c) @ e2
    span = max(float(torch.quantile(torch.stack([p1.abs(), p2.abs()]), 0.98)),
               float(torch.stack([q1.abs(), q2.abs()]).max())) * args.span
    print('feature plane: %d images projected, half-width %.2f' % (seen, span))

    g = args.grid
    ts = torch.linspace(-span, span, g, dtype=torch.float64)
    A, B = torch.meshgrid(ts, ts, indexing='ij')
    grid = c.unsqueeze(0) + A.reshape(-1, 1) * e1 + B.reshape(-1, 1) * e2   # CPU
    with torch.no_grad():
        logits = head(grid.float().to(dev)).cpu()                           # device
    cell = logits.argmax(1).reshape(g, g)
    # height: the winning score negated, i.e. the lower envelope min_k g_k
    Z = (-logits.max(1).values).reshape(g, g)

    # write the projected data too, so the figure can show where images live
    with open(args.out.replace('.csv', '_points.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['x', 'y', 'label'])
        for u, v, l in zip(p1.tolist(), p2.tolist(), Yf.tolist()):
            wr.writerow([round(u, 5), round(v, 5), l])
    return ts, ts, Z, cell


# ----------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--mode', default='feature', choices=['param', 'feature'])
    ap.add_argument('--scope', default='all',
                    help="param mode: all | backbone | head | a substring of layer names")
    ap.add_argument('--grid', type=int, default=None)
    ap.add_argument('--center', default='0,0',
                    help="param mode: centre of the window in slice coordinates, e.g. 0.7,0.7. "
                         "The plane is unchanged, so a zoom lines up with the full slice.")
    ap.add_argument('--span', type=float, default=1.0)
    ap.add_argument('--images', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--split', default='test', choices=['test', 'train'],
                    help='images the loss is evaluated on (train: without augmentation, fixed order)')
    ap.add_argument('--plane', default='pca',
                    help="feature mode: 'pca' for the top-two prototype directions, "
                         "or three class indices like 0,1,2 for the plane through "
                         "those three prototypes")
    ap.add_argument('--fake-data', action='store_true',
                    help='use random images instead of the dataset (testing only)')
    args = ap.parse_args()
    torch.manual_seed(args.seed)          # reproducible directions and test data
    args.center = tuple(float(v) for v in args.center.split(','))
    if args.grid is None:
        args.grid = 21 if args.mode == 'param' else 201

    if args.device:
        dev = torch.device(args.device)
    elif torch.cuda.is_available():
        dev = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    run = args.ckpt.split('/')[-2] if '/' in args.ckpt else args.ckpt
    if args.out is None:
        args.out = os.path.join('results', 'slices', '%s_%s_%s%s.csv' % (
            run, args.mode, args.scope if args.mode == 'param' else 'plane',
            '_train' if args.split == 'train' else ''))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print('%s | %s head | mode %s | device %s' %
          (run, ck['args']['head'], args.mode, dev))

    X, Y, Z, cell = (param_surface if args.mode == 'param' else feature_surface)(ck, dev, args)

    with open(args.out, 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['x', 'y', 'z'] + (['cell'] if cell is not None else []))
        for i in range(len(X)):
            for j in range(len(Y)):
                row = [round(float(X[i]), 6), round(float(Y[j]), 6), round(float(Z[i, j]), 6)]
                if cell is not None:
                    row.append(int(cell[i, j]))
                wr.writerow(row)
    print('wrote', args.out)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        Xg, Yg = torch.meshgrid(X, Y, indexing='ij')
        fig = plt.figure(figsize=(11, 4.6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_wireframe(Xg.numpy(), Yg.numpy(), Z.numpy(), rstride=max(1, len(X)//40),
                          cstride=max(1, len(X)//40), linewidth=0.4, color='0.25')
        ax.set_xlabel('direction 1'); ax.set_ylabel('direction 2')
        ax.set_zlabel('loss' if cell is None else 'winning score')
        ax.set_title(run + ('  (parameter slice)' if cell is None else '  (feature plane)'))
        ax2 = fig.add_subplot(1, 2, 2)
        if cell is not None:
            ax2.imshow(cell.numpy().T, origin='lower', cmap='tab10',
                       extent=[X[0], X[-1], Y[0], Y[-1]], interpolation='nearest')
            ax2.set_title('decision cells on this plane')
        else:
            cs = ax2.contour(Xg.numpy(), Yg.numpy(), Z.numpy(), 30, linewidths=0.6)
            ax2.clabel(cs, inline=True, fontsize=6); ax2.set_title('contours')
        ax2.set_xlabel('direction 1'); ax2.set_ylabel('direction 2')
        plt.tight_layout(); plt.savefig(args.out.replace('.csv', '.png'), dpi=130)
        print('wrote', args.out.replace('.csv', '.png'))
    except ImportError:
        print('(matplotlib not installed; CSV only)')


if __name__ == '__main__':
    main()
