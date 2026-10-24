"""
probe.py -- measurements on trained checkpoints (Sections 5 and 7).

    python3 src/probe.py --selftest                       about a minute, no data needed
    python3 src/probe.py "runs/S_*/final.pt" "runs/G_*/final.pt" --out results/probe

Keep the quotes around each pattern: the script expands them itself, and a pattern
that matches nothing is reported and skipped. Nothing is trained and nothing in runs/
is changed. Each checkpoint is loaded once and its features are computed once (test
set, training set without augmentation, one augmented pass of the training set);
everything else is computed from those features in double precision on the CPU.

Writes three files into --out (default results/probe):

  probe_runs.csv        one row per run
      accuracy, NLL and expected calibration error (15 bins) on the test set;
      mean feature norm on test and training features;
      mean logit gap between the two top classes, the norm of its gradient with
      respect to the feature ("gain"), and norm x gain;
      fraction of correctly classified test features that stay in their class far
      out along their own ray ("escape", Section 7.1);
      head scale: |w|, alpha, tau, |a|, spread of rho (conic); |W|, spread of b
      (affine); learned scale (cosine); aperture spread (proto);
      for conic and proto heads: dL/dtau and the weight-decay force on eta (for the
      proto head the dLdtau columns hold dL/dalpha, since alpha = softplus(eta));
      out-of-distribution test: AUROC of the maximum softmax probability and of the
      feature norm, with the test set of the other CIFAR as outside data (the paper
      uses this one); --svhn adds the SVHN test set (needs scipy, downloads 64 MB);
      --no-ood skips the test.

  probe_classes.csv     one row per class of every conic, proto and affine head
      asymptotic margin M_k (Proposition 7) as an interval [M_lo, M_hi] that brackets
      the exact value; distance from w_k to the convex hull of the other axes and the
      bound dist + min_j (alpha_j - alpha_k) of Corollary 8; dL/deta_k, dL/dtau_k and
      the weight-decay force on eta_k.

  probe_finallayer.csv  affine heads only (Proposition 10)
      band width m_+ - m_- of eq. (sandwich), anchored at the trained weights and at
      W = 0; the recession function L_inf on training and test features along the
      trained direction W0, along the 72 rays of the plane of Fig. 2 (the directions
      of loss_surface.py --scope head --seed 0), and along 200 further
      filter-normalised random directions.
"""
from __future__ import annotations
import argparse, csv, glob, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import make_backbone, make_head, get_data, CIFAR_MEAN, CIFAR_STD

DT = torch.float64


# ------------------------------------------------------------------ loading
def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def rebuild(ck):
    a = ck['args']
    K = 10 if a.get('dataset', 'cifar10') == 'cifar10' else 100
    net = make_backbone(a.get('backbone', 'resnet18'), not a.get('no_bn', False),
                        not a.get('no_skip', False))
    head = make_head(a['head'], net.feat_dim, K, a.get('eps', 0.1), a.get('tau_init', 1.5),
                     not a.get('fix_tau', False), a.get('tau_floor', 0.0))
    net.load_state_dict(ck['net']); head.load_state_dict(ck['head'])
    return net.eval(), head.eval(), a, K


# --------------------------------------------------------------------- data
def loaders(dataset, data_dir, fake):
    """test, train without augmentation, train with the training augmentation."""
    if fake:
        g = torch.Generator().manual_seed(0)
        mk = lambda n: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)),
            batch_size=256, shuffle=False)
        return mk(512), mk(512), mk(512)
    from torchvision import datasets, transforms
    norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    plain = transforms.Compose([transforms.ToTensor(), norm])
    aug = transforms.Compose([transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    cls = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}[dataset]
    mk = lambda train, tf: torch.utils.data.DataLoader(
        cls(data_dir, train=train, download=True, transform=tf), batch_size=500, shuffle=False)
    return mk(False, plain), mk(True, plain), mk(True, aug)


def other_cifar_loader(dataset, data_dir, fake):
    """Test set of the other CIFAR, used as outside data."""
    if fake:
        g = torch.Generator().manual_seed(2)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.randn(512, 3, 32, 32, generator=g) * 1.2, torch.zeros(512, dtype=torch.long)),
            batch_size=256)
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    cls = datasets.CIFAR100 if dataset == 'cifar10' else datasets.CIFAR10
    return torch.utils.data.DataLoader(cls(data_dir, train=False, download=True, transform=tf),
                                       batch_size=500)


def svhn_loader(data_dir, fake):
    if fake:
        g = torch.Generator().manual_seed(1)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.randn(512, 3, 32, 32, generator=g) * 1.5, torch.zeros(512, dtype=torch.long)),
            batch_size=256)
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return torch.utils.data.DataLoader(
        datasets.SVHN(data_dir, split='test', download=True, transform=tf), batch_size=500)


@torch.no_grad()
def features(net, loader, dev):
    Z, Y = [], []
    for x, y in loader:
        Z.append(net(x.to(dev)).cpu().to(DT)); Y.append(y)
    return torch.cat(Z), torch.cat(Y)


# ------------------------------------------------------------- head algebra
def head64(head):
    """A double-precision copy of the head, on the CPU."""
    import copy
    h = copy.deepcopy(head).cpu().to(DT)
    for p in h.parameters():
        p.requires_grad_(False)
    return h


def ray_coefficients(kind, h):
    """Score g_k - rho_k ~ t c_k(u) + e_k(u) along z = t u, |u| = 1, t -> inf.
    Returns q (K,d), beta (K,) with c_k(u) = <q_k,u> + beta_k, and a function
    giving the next-order term e_k(u) for tie-breaking."""
    if kind == 'conic':
        al = h.alpha(); q, beta = h.w, al
        e = lambda U: -(h.w * h.a).sum(1)[None, :] - al[None, :] * (U @ h.a.t()) - h.rho[None, :]
    elif kind == 'proto':
        al = h.alpha(); q, beta = torch.zeros_like(h.a), al
        e = lambda U: -al[None, :] * (U @ h.a.t()) - h.rho[None, :]
    elif kind == 'affine':
        q, beta = -h.fc.weight, torch.zeros(h.fc.weight.shape[0], dtype=DT)
        e = lambda U: -h.fc.bias[None, :].expand(U.shape[0], -1)
    else:
        return None
    return q, beta, e


def escape_fraction(kind, h, Z, Y, pred):
    """Share of correctly classified features z for which mu z stays in the class
    of z for every sufficiently large mu."""
    ok = pred == Y
    if kind == 'cosine':
        return 1.0                         # logits do not depend on |z|
    rc = ray_coefficients(kind, h)
    if rc is None or int(ok.sum()) == 0:
        return float('nan')
    q, beta, e = rc
    U = F.normalize(Z[ok], dim=1)
    c = U @ q.t() + beta[None, :]
    tol = 1e-9 * (c.abs().max(1, keepdim=True).values + 1.0)
    tied = c <= c.min(1, keepdim=True).values + tol
    ee = e(U).masked_fill(~tied, float('inf'))
    return float((ee.argmin(1) == Y[ok]).to(DT).mean())


def simplex_proj(v):
    u, _ = torch.sort(v, descending=True)
    css = u.cumsum(0) - 1
    idx = torch.arange(1, v.numel() + 1, dtype=v.dtype)
    r = int((u - css / idx > 0).nonzero().max())
    return (v - css[r] / (r + 1)).clamp_min(0)


def dual_margin(Vm, b, iters=2000):
    """min over the simplex of |V^T lam| + b^T lam  (V: m x d), by FISTA on a
    smoothed norm with continuation. Returns (lower, upper, |V^T lam|).
    upper = objective at the final lam. lower = the best primal value
    min_j(<v_j,u> + b_j) over the feasible points u (|u| <= 1) met on the way:
    the smoothed problem min sqrt(|V^T lam|^2 + delta^2) + b^T lam is the dual of
    max_{|u|^2+s^2<=1} min_j(<v_j,u> + b_j) + s delta, whose optimal u is
    r / sqrt(|r|^2 + delta^2) with r = V^T lam; it lies in the unit ball."""
    m = Vm.shape[0]
    s = float(Vm.norm(dim=1).max()) + float(b.abs().max()) + 1e-30
    lam = torch.full((m,), 1.0 / m, dtype=DT)
    L0 = float(torch.linalg.matrix_norm(Vm, 2)) ** 2
    cands = [float(b.min())] if Vm.shape[1] > m else []
    prim = lambda u: float((Vm @ u + b).min())
    for delta in (1e-1 * s, 1e-2 * s, 1e-3 * s, 1e-4 * s, 1e-6 * s, 1e-8 * s):
        L = L0 / delta + 1e-12
        y, t, prev = lam.clone(), 1.0, lam.clone()
        for _ in range(iters):
            r = Vm.t() @ y
            gr = Vm @ r / torch.sqrt(r @ r + delta ** 2) + b
            new = simplex_proj(y - gr / L)
            tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
            y = new + ((t - 1) / tn) * (new - prev)
            prev, t = new, tn
        lam = prev
        r = Vm.t() @ lam
        cands.append(prim(r / torch.sqrt(r @ r + delta ** 2)))
        if float(r.norm()) > 0:
            cands.append(prim(r / r.norm()))
    r = Vm.t() @ lam
    upper = float(r.norm() + b @ lam)
    return max(cands), upper, float(r.norm())


def margins(kind, h):
    rc = ray_coefficients(kind, h)
    if rc is None:
        return []
    q, beta, _ = rc
    K, d = q.shape
    rows = []
    for k in range(K):
        oth = [j for j in range(K) if j != k]
        Vm = q[oth] - q[k]
        bb = beta[oth] - beta[k]
        lo, hi, _ = dual_margin(Vm, bb)
        dlo, dhi, dist = dual_margin(Vm, torch.zeros_like(bb))    # dist(q_k, conv q_j)
        rows.append(dict(k=k, M_lo=lo, M_hi=hi, dist_conv=dhi,
                         bound=dhi + float(bb.min()), min_dbeta=float(bb.min()),
                         q_norm=float(q[k].norm())))
    return rows


def eta_gradients(kind, h, Z, Y, a):
    """dL/deta for the mean cross-entropy over the given features, with every
    other parameter held fixed, and the weight-decay force lambda*eta."""
    if kind not in ('conic', 'proto'):
        return None
    eta = h.eta.detach().clone().requires_grad_(True)
    h.eta = torch.nn.Parameter(eta)            # temporarily differentiable
    loss = 0.0
    for i in range(0, Z.shape[0], 5000):
        loss = loss + F.cross_entropy(h(Z[i:i + 5000]), Y[i:i + 5000], reduction='sum')
    loss = loss / Z.shape[0]
    g, = torch.autograd.grad(loss, h.eta)
    h.eta.requires_grad_(False)
    lam = a.get('wd_eta', None)
    lam = a.get('wd', 5e-4) if lam is None else lam
    eta0 = h.eta.detach()
    dtau = g / torch.sigmoid(eta0)             # tau = floor + softplus(eta)
    return dict(grad_eta=g, grad_tau=dtau, wd_force=lam * eta0, lam=lam, eta=eta0)


def gap_and_gain(h, Z):
    """Top-1 minus top-2 logit, and the norm of its gradient w.r.t. z."""
    Zg = Z.clone().requires_grad_(True)
    s = h(Zg)
    top = s.topk(2, dim=1)
    gap = top.values[:, 0] - top.values[:, 1]
    g, = torch.autograd.grad(gap.sum(), Zg)
    return gap.detach(), g.norm(dim=1)


def ece(prob, Y, bins=15):
    conf, pred = prob.max(1)
    acc = (pred == Y).to(DT)
    e = torch.zeros((), dtype=DT)
    edges = torch.linspace(0, 1, bins + 1, dtype=DT)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.to(DT).mean() * (conf[m].mean() - acc[m].mean()).abs()
    return float(e)


def auroc(pos, neg):
    """P(score of an in-distribution sample > score of an OOD sample)."""
    x = torch.cat([pos, neg]); r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(1, x.numel() + 1, dtype=DT)
    n1, n0 = pos.numel(), neg.numel()
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------- final layer (Prop. 10)
def l_inf(D, Z, Y):
    q = Z @ D.t()
    return float((q.max(1).values - q.gather(1, Y[:, None]).squeeze(1)).mean())


def band(S, Y, K):
    sy = S.gather(1, Y[:, None]).squeeze(1)
    m_lo = float((S.min(1).values - sy).mean())
    m_hi = float((S.max(1).values - sy).mean()) + math.log(K)
    return m_lo, m_hi


def final_layer_rows(name, h, K, sets):
    W0, b = h.fc.weight.detach(), h.fc.bias.detach()
    gen = torch.Generator().manual_seed(0)          # as loss_surface.py --seed 0
    fn = lambda r: r * (W0.norm(dim=1, keepdim=True) / (r.norm(dim=1, keepdim=True) + 1e-12))
    D1 = fn(torch.randn(W0.shape, generator=gen, dtype=torch.float32).to(DT))
    D2 = fn(torch.randn(W0.shape, generator=gen, dtype=torch.float32).to(DT))
    g2 = torch.Generator().manual_seed(12345)
    extra = [fn(torch.randn(W0.shape, generator=g2, dtype=torch.float32).to(DT)) for _ in range(200)]
    rows = []
    for split, (Z, Y) in sets.items():
        S0 = Z @ W0.t() + b
        loss0 = float(F.cross_entropy(S0, Y))
        lo, hi = band(S0, Y, K)
        lo0, hi0 = band(b.expand(Z.shape[0], -1), Y, K)
        ang = [l_inf(D1 * math.cos(t) + D2 * math.sin(t), Z, Y)
               for t in [2 * math.pi * i / 72 for i in range(72)]]
        rnd = [l_inf(D, Z, Y) for D in extra]
        zero = lambda v: sum(1 for x in v if x < 1e-9)
        rows.append(dict(
            run=name, split=split, n=Z.shape[0], loss_W0=loss0,
            acc_W0=float((S0.argmax(1) == Y).to(DT).mean()),
            band_W0=hi - lo, m_lo_W0=lo, m_hi_W0=hi, band_zero=hi0 - lo0,
            Linf_trained_dir=l_inf(W0 / W0.norm(), Z, Y),
            plane_min=min(ang), plane_median=sorted(ang)[36], plane_max=max(ang),
            plane_zero_rays=zero(ang),
            rand_min=min(rnd), rand_median=sorted(rnd)[100], rand_max=max(rnd),
            rand_zero_dirs=zero(rnd)))
    return rows


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--out', default='results/probe')
    ap.add_argument('--device', default=None)
    ap.add_argument('--no-ood', action='store_true', help='skip the out-of-distribution test')
    ap.add_argument('--svhn', action='store_true',
                    help='also use the SVHN test set as outside data (needs scipy)')
    ap.add_argument('--selftest', action='store_true', help='one-minute check, no data needed')
    ap.add_argument('--fake-data', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    run_probe(args)


def run_probe(args):
    paths = []
    for pat in args.ckpts:
        hit = sorted(glob.glob(pat))
        if not hit:
            print(f'  note: nothing matches {pat} -- skipped')
        paths += [p for p in hit if p not in paths]
    if not paths:
        raise SystemExit('no checkpoints matched')
    svhn_ok = False
    if getattr(args, 'svhn', False):
        try:
            import scipy  # noqa: F401  (torchvision needs it to read SVHN)
            svhn_ok = True
        except Exception:
            print('  note: --svhn needs scipy; SVHN skipped')
    dev = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    print(f'device {dev} | {len(paths)} checkpoints -> {args.out}/')

    run_rows, class_rows, fl_rows = [], [], []
    cache, ood_cache = {}, {}
    for path in paths:
        try:
            t0 = time.time()
            name = os.path.basename(os.path.dirname(path)) or path
            ck = torch.load(path, map_location='cpu', weights_only=False)
            net, head, a, K = rebuild(ck)
            kind = a['head']
            net = net.to(dev)
            ds = a.get('dataset', 'cifar10')
            if ds not in cache:
                cache[ds] = loaders(ds, args.data_dir, args.fake_data)
            te_ld, tr_ld, au_ld = cache[ds]
            torch.manual_seed(0)                       # the same augmented pass for every run
            Zte, Yte = features(net, te_ld, dev)
            Ztr, Ytr = features(net, tr_ld, dev)
            Zau, Yau = features(net, au_ld, dev)
            h = head64(head)

            with torch.no_grad():
                s_te = h(Zte)
            prob = torch.softmax(s_te, 1)
            pred = s_te.argmax(1)
            gap, gain = gap_and_gain(h, Zte)
            nz = Zte.norm(dim=1)
            row = dict(run=name, head=kind, bn='no' if a.get('no_bn') else 'yes',
                       lr=a.get('lr'), eps=a.get('eps'), seed=a.get('seed'), dataset=ds,
                       wd=a.get('wd', 5e-4), wd_eta=a.get('wd_eta', a.get('wd', 5e-4)),
                       tau_floor=a.get('tau_floor', 0.0),
                       acc=float((pred == Yte).to(DT).mean()),
                       nll=float(F.cross_entropy(s_te, Yte)), ece=ece(prob, Yte),
                       mean_maxprob=float(prob.max(1).values.mean()),
                       fnorm_test=float(nz.mean()), fnorm_train=float(Ztr.norm(dim=1).mean()),
                       fnorm_train_aug=float(Zau.norm(dim=1).mean()),
                       gap=float(gap.mean()), gain=float(gain.mean()),
                       norm_x_gain=float((nz * gain).mean()),
                       escape=escape_fraction(kind, h, Zte, Yte, pred))
            if kind == 'conic':
                al, wn = h.alpha(), h.w.norm(dim=1)
                row.update(tau=float(h.tau().mean()), w_norm=float(wn.mean()), alpha=float(al.mean()),
                           a_norm=float(h.a.norm(dim=1).mean()), rho_sd=float(h.rho.std()),
                           alpha_spread=float((al.max() - al.min()) / wn.mean()),
                           eps_rel=float(a.get('eps', 0.1) / (Zte - h.a[Yte]).norm(dim=1).mean()))
            elif kind == 'proto':
                al = h.alpha()
                row.update(alpha=float(al.mean()), alpha_cv=float(al.std() / al.mean()),
                           alpha_spread=float((al.max() - al.min()) / al.mean()),
                           a_norm=float(h.a.norm(dim=1).mean()), rho_sd=float(h.rho.std()))
            elif kind == 'affine':
                row.update(w_norm=float(h.fc.weight.norm(dim=1).mean()), b_sd=float(h.fc.bias.std()))
            elif kind == 'cosine':
                row.update(cos_scale=float(h.scale))

            grads = {}
            for split, (Z, Y) in (('train', (Ztr, Ytr)), ('train_aug', (Zau, Yau))):
                g = eta_gradients(kind, h, Z, Y, a)
                if g is not None:
                    grads[split] = g
                    row[f'dLdtau_{split}'] = float(g['grad_tau'].mean())
                    row[f'n_dLdtau_neg_{split}'] = int((g['grad_tau'] < 0).sum())
                    row[f'wd_force_{split}'] = float(g['wd_force'].mean())
                    row[f'dLdeta_{split}'] = float(g['grad_eta'].mean())

            if not args.no_ood:
                if ds not in ood_cache:
                    ood_cache[ds] = {('cifar100' if ds == 'cifar10' else 'cifar10'):
                                     other_cifar_loader(ds, args.data_dir, args.fake_data)}
                    if svhn_ok:
                        try:
                            ood_cache[ds]['svhn'] = svhn_loader(args.data_dir, args.fake_data)
                        except Exception as e:
                            print(f'  SVHN skipped ({e.__class__.__name__}); the other tests continue.')
                for oname, old in ood_cache[ds].items():
                    Zo, _ = features(net, old, dev)
                    with torch.no_grad():
                        po = torch.softmax(h(Zo), 1)
                    row[f'auroc_msp_{oname}'] = auroc(prob.max(1).values, po.max(1).values)
                    row[f'auroc_fnorm_{oname}'] = auroc(nz, Zo.norm(dim=1))
                    row[f'fnorm_{oname}'] = float(Zo.norm(dim=1).mean())

            for r in margins(kind, h):
                r.update(run=name, head=kind)
                if kind in ('conic', 'proto'):
                    k = r['k']
                    r['alpha_k'] = float(h.alpha()[k])
                    if kind == 'conic':
                        r['tau_k'] = float(h.tau()[k]); r['w_norm_k'] = float(h.w[k].norm())
                    for split, g in grads.items():
                        r[f'dLdeta_{split}'] = float(g['grad_eta'][k])
                        r[f'dLdtau_{split}'] = float(g['grad_tau'][k])
                        r[f'wd_force_{split}'] = float(g['wd_force'][k])
                class_rows.append(r)
            if class_rows:
                mine = [r for r in class_rows if r['run'] == name]
                if mine:
                    row['M_min_lo'] = min(r['M_lo'] for r in mine)
                    row['M_gap_max'] = max(r['M_hi'] - r['M_lo'] for r in mine)
                    row['n_open'] = sum(1 for r in mine if r['M_lo'] > 0)
                    row['n_closed'] = sum(1 for r in mine if r['M_hi'] < 0)

            if kind == 'affine':
                fl_rows += final_layer_rows(name, h, K, {'train': (Ztr, Ytr), 'test': (Zte, Yte)})

            run_rows.append(row)
            print(f'  {name:28s} {kind:6s} acc {row["acc"]:.4f}  |z| {row["fnorm_test"]:7.3f}  '
                  f'ece {row["ece"]:.4f}  escape {row["escape"]:.4f}'
                  + (f'  open {row.get("n_open")}/{K}' if 'n_open' in row else '')
                  + (f'  dL/d{"alpha" if kind == "proto" else "tau"} {row["dLdtau_train"]:+.2e}'
                     if 'dLdtau_train' in row else '')
                  + f'  ({time.time() - t0:.0f} s)')
        except Exception as e:                      # one bad checkpoint does not stop the rest
            print(f'  {path}: skipped ({e.__class__.__name__}: {e})')
            class_rows = [r for r in class_rows if r.get("run") != os.path.basename(os.path.dirname(path))]
            fl_rows = [r for r in fl_rows if r.get("run") != os.path.basename(os.path.dirname(path))]

    def dump(rows, fname):
        if not rows:
            return
        cols = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        with open(os.path.join(args.out, fname), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows:
                w.writerow({c: (f'{v:.6g}' if isinstance(v, float) else v) for c, v in r.items()})
        print(f'wrote {args.out}/{fname} ({len(rows)} rows)')

    dump(run_rows, 'probe_runs.csv')
    dump(class_rows, 'probe_classes.csv')
    dump(fl_rows, 'probe_finallayer.csv')
    if len(run_rows) < len(paths):
        print(f'{len(paths) - len(run_rows)} of {len(paths)} checkpoints were skipped (see above).')
    if not run_rows:
        raise SystemExit('PROBE FAILED: no checkpoint could be measured.')
    print('PROBE DONE.')
    return run_rows, class_rows, fl_rows


def selftest():
    """Builds three small random checkpoints (conic, affine, proto) in a temporary
    folder, runs the whole probe on random images, and checks the margin solver."""
    import tempfile, types
    ok = True
    # 1. the margin solver on a case with a known answer: the simplex frame of
    #    neural collapse, where dist(w_0, conv{w_j}) = K/(K-1) |w|.
    K, d = 10, 64
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(d, K, generator=g, dtype=DT))
    W = 2.0 * ((torch.eye(K, dtype=DT) - 1.0 / K) @ Q.t()) * math.sqrt(K / (K - 1))
    lo, hi, _ = dual_margin(W[1:] - W[0], torch.zeros(K - 1, dtype=DT), iters=800)
    want = K / (K - 1) * float(W[0].norm())
    print(f'  margin solver: {lo:.6f} <= {want:.6f} <= {hi:.6f}')
    ok &= lo - 1e-6 <= want <= hi + 1e-6
    # 2. the whole pipeline on random checkpoints
    with tempfile.TemporaryDirectory() as tmp:
        for kind in ('conic', 'affine', 'proto'):
            torch.manual_seed(1)
            a = dict(dataset='cifar10', backbone='resnet18', head=kind, no_bn=False,
                     no_skip=False, eps=0.5, tau_init=1.5, fix_tau=False, tau_floor=0.0,
                     lr=0.1, seed=0, wd=5e-4, wd_eta=5e-4)
            net = make_backbone('resnet18', True, True)
            head = make_head(kind, net.feat_dim, 10, 0.5, 1.5, True, 0.0)
            os.makedirs(os.path.join(tmp, 'runs', f'ST_{kind}'))
            torch.save({'net': net.state_dict(), 'head': head.state_dict(), 'args': a},
                       os.path.join(tmp, 'runs', f'ST_{kind}', 'final.pt'))
        args = types.SimpleNamespace(
            ckpts=[os.path.join(tmp, 'runs', 'ST_*', 'final.pt')], data_dir='./data',
            out=os.path.join(tmp, 'out'), device=None, no_ood=False, fake_data=True)
        runs, classes, fl = run_probe(args)
        ok &= len(runs) == 3 and len(fl) == 2 and len(classes) == 30
        ok &= max(r['M_hi'] - r['M_lo'] for r in classes) < 1e-5
        ok &= all(os.path.exists(os.path.join(tmp, 'out', f)) for f in
                  ('probe_runs.csv', 'probe_classes.csv', 'probe_finallayer.csv'))
    print('SELFTEST PASSED' if ok else 'SELFTEST FAILED')
    return ok


if __name__ == '__main__':
    main()
