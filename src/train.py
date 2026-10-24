"""
train.py -- one training run: dataset -> backbone -> head -> SGD -> log.

    python3 src/train.py --smoke                      install check, 2 epochs on random data
    python3 src/train.py --dataset cifar10 --backbone resnet18 --head conic --eps 0.5 --out NAME

The backbones are the CIFAR variants of ResNet-18 and ResNet-56 (3x3 stem, no
max-pool). The heads are affine, cosine, conic (eq. (1)) and proto (the prototype
limit of the conic head, w_k = 0). Training uses SGD with momentum 0.9, weight decay
5e-4, batch 128 and a cosine schedule over --epochs.

Outputs in runs/<NAME>/ (or <--runs-dir>/<NAME>/):
    log.csv      one row per epoch (accuracy, loss, feature norm, tau, ...)
    config.json  every option of the run
    final.pt     weights and options (omitted with --no-final)
    DONE         written when the run has finished (or diverged), so drivers can skip it
    DIVERGED     written when the loss became non-finite

An interrupted run continues from its last completed epoch when the same command is
given again (ckpt.pt holds the state until the run finishes).
"""

from __future__ import annotations
import argparse, csv, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from conic_head import ConicHead, ProtoHead


# ===========================================================================
# 1. BACKBONES. CIFAR variants: a 3x3 first convolution instead of 7x7 and no
#    initial max-pool, as is standard for 32x32 images.
# ===========================================================================

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, use_bn, use_skip=True):
        super().__init__()
        norm = nn.BatchNorm2d if use_bn else (lambda c: nn.Identity())
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=not use_bn)
        self.n1 = norm(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=not use_bn)
        self.n2 = norm(cout)
        self.use_skip = use_skip
        self.short = nn.Sequential()
        # without a skip path there is nothing to project, so no shortcut
        # parameters are created at all: the network is then a plain stack
        if use_skip and (stride != 1 or cin != cout):
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=not use_bn), norm(cout))

    def forward(self, x):
        out = F.relu(self.n1(self.c1(x)))
        out = self.n2(self.c2(out))
        if self.use_skip:
            out = out + self.short(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """CIFAR variant. Outputs a 512-dimensional feature, not logits."""
    feat_dim = 512

    def __init__(self, use_bn=True):
        super().__init__()
        norm = nn.BatchNorm2d if use_bn else (lambda c: nn.Identity())
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=not use_bn),
                                  norm(64), nn.ReLU(inplace=True))
        layers, cin = [], 64
        for cout, stride in [(64, 1), (64, 1), (128, 2), (128, 1),
                             (256, 2), (256, 1), (512, 2), (512, 1)]:
            layers.append(BasicBlock(cin, cout, stride, use_bn)); cin = cout
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.blocks(self.stem(x))).flatten(1)


class ResNet56(nn.Module):
    """The CIFAR ResNet-56 of He et al. (2016): a 3x3 stem, then three stages of
    nine basic blocks with 16, 32 and 64 channels, 55 convolutions in all. Thin
    and deep, which is the regime where Li et al. (2018) found that removing the
    skip connections turns a nearly convex loss surface into a chaotic one.
    Outputs a 64-dimensional feature."""
    feat_dim = 64

    def __init__(self, use_bn=True, use_skip=True, n=9):
        super().__init__()
        norm = nn.BatchNorm2d if use_bn else (lambda c: nn.Identity())
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=not use_bn),
                                  norm(16), nn.ReLU(inplace=True))
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for b in range(n):
                layers.append(BasicBlock(cin, cout, stride if b == 0 else 1,
                                         use_bn, use_skip))
                cin = cout
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():                 # He initialisation, as in the
            if isinstance(m, nn.Conv2d):         # original CIFAR recipe
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.pool(self.blocks(self.stem(x))).flatten(1)


BACKBONES = {'resnet18': ResNet18, 'resnet56': ResNet56}


def make_backbone(name, use_bn=True, use_skip=True):
    """Single place that builds a backbone, so that training and every analysis
    script construct the identical network."""
    if name == 'resnet56':
        return ResNet56(use_bn=use_bn, use_skip=use_skip)
    if not use_skip:
        raise SystemExit('--no-skip is implemented only for --backbone resnet56')
    return BACKBONES[name](use_bn=use_bn)


# ===========================================================================
# 2. HEADS. Each maps a feature to K logits.
# ===========================================================================

class AffineHead(nn.Module):
    """The standard head: a single linear layer."""
    def __init__(self, dim, K):
        super().__init__()
        self.fc = nn.Linear(dim, K)

    def forward(self, z):
        return self.fc(z)


class CosineHead(nn.Module):
    """Normalised (cosine) head with a learned scale that starts at 16. Its logits
    do not depend on the feature norm."""
    def __init__(self, dim, K, scale=16.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(K, dim) / math.sqrt(dim))
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, z):
        return self.scale * F.normalize(z, dim=1) @ F.normalize(self.w, dim=1).t()


def make_head(name, dim, K, eps, tau_init, learn_tau, tau_floor=0.0):
    if name == 'proto':
        return ProtoHead(dim, K, eps=eps, alpha_init=tau_init)
    if name == 'affine':
        return AffineHead(dim, K)
    if name == 'cosine':
        return CosineHead(dim, K)
    if name == 'conic':
        return ConicHead(dim, K, eps=eps, tau_init=tau_init, learn_tau=learn_tau,
                         tau_floor=tau_floor)
    raise ValueError(f'unknown head: {name}')


# ===========================================================================
# 3. DATA.  torchvision downloads CIFAR automatically on first use.
# ===========================================================================

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def get_data(dataset, data_dir, batch, workers, dev, fake=False):
    if fake:                                   # random images, same number of classes
        n, K = 512, (10 if dataset == 'cifar10' else 100)
        x = torch.randn(n, 3, 32, 32); y = torch.randint(0, K, (n,))
        ds = torch.utils.data.TensorDataset(x, y)
        ld = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
        return ld, ld, K

    from torchvision import datasets, transforms
    mean, std = CIFAR_MEAN, CIFAR_STD
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    cls = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}[dataset]
    K = 10 if dataset == 'cifar10' else 100
    tr = cls(data_dir, train=True, download=True, transform=train_tf)
    te = cls(data_dir, train=False, download=True, transform=test_tf)
    # pin_memory is CUDA-only. persistent_workers keeps the loader processes alive
    # across epochs (macOS otherwise runs out of file descriptors).
    kw = dict(num_workers=workers,
              pin_memory=(dev.type == 'cuda'),
              persistent_workers=(workers > 0))
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True, drop_last=True, **kw),
            torch.utils.data.DataLoader(te, 256, shuffle=False, **kw), K)


# ===========================================================================
# 4. TRAINING
# ===========================================================================

def evaluate(net, head, loader, dev):
    net.eval(); head.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            correct += (head(net(x)).argmax(1) == y).sum().item(); total += y.numel()
    net.train(); head.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description='Conic head experiment runner')
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    ap.add_argument('--backbone', default='resnet18', choices=list(BACKBONES))
    ap.add_argument('--head', default='conic', choices=['affine', 'cosine', 'conic', 'proto'])
    ap.add_argument('--no-bn', action='store_true', help='remove all BatchNorm layers')
    ap.add_argument('--no-skip', action='store_true',
                    help='remove the skip connections (resnet56 only): a plain deep stack')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--eps', type=float, default=0.1, help='conic head smoothing')
    ap.add_argument('--tau-init', type=float, default=1.5, help='initial signature ratio')
    ap.add_argument('--fix-tau', action='store_true', help='do not learn tau')
    ap.add_argument('--diag-every', type=int, default=100,
                    help='batches between closed-form curvature measurements of the '
                         'conic head (column head_curv); 0 turns them off')
    ap.add_argument('--diag-samples', type=int, default=32,
                    help='samples used per curvature measurement')
    ap.add_argument('--tau-freeze', type=int, default=0,
                    help='hold tau fixed for the first N epochs, then release it')
    ap.add_argument('--tau-floor', type=float, default=0.0,
                    help='lower bound on tau; 1.0 keeps every class region an ellipsoid')
    ap.add_argument('--wd-eta', type=float, default=None,
                    help='weight decay on eta (tau of the conic head, aperture of the '
                         'proto head); default: the same as --wd. Use 0 to take tau '
                         'out of weight decay.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--out', default=None, help='run name; auto-named if omitted')
    ap.add_argument('--runs-dir', default=os.environ.get('RUNS_DIR', 'runs'),
                    help='folder that holds the run folders (default runs, or $RUNS_DIR)')
    ap.add_argument('--workers', type=int, default=None,
                    help='data-loading processes; default 4 on CUDA, 0 elsewhere')
    ap.add_argument('--restart', action='store_true',
                    help='ignore any existing checkpoint and start from epoch 0')
    ap.add_argument('--no-final', action='store_true',
                    help='do not save final.pt (sweeps where only the log matters)')
    ap.add_argument('--keep-ckpt', action='store_true',
                    help='keep the resume checkpoint after a run finishes '
                         '(it is ~2x the size of final.pt and no longer needed)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true',
                    help='2 epochs on random data: checks the install, downloads nothing')
    ap.add_argument('--fake-data', action='store_true',
                    help='random images instead of CIFAR (testing only)')
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.fake_data, args.workers = 2, True, 0

    torch.manual_seed(args.seed)
    if args.device:
        dev = torch.device(args.device)
    elif torch.cuda.is_available():
        dev = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')

    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True   # fastest convolution algorithms for fixed shapes

    # Defaults that depend on the device. On Apple MPS the device, not data
    # loading, is the bottleneck, so worker processes buy nothing there.
    if args.workers is None:
        args.workers = 4 if dev.type == 'cuda' else 0
    if args.workers > 0:
        try:                                   # raise the open-file limit (macOS: 256)
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, 8192)), hard))
        except Exception:
            pass
        if sys.platform == 'darwin':           # macOS only: avoids running out of
            try:                               # file descriptors
                torch.multiprocessing.set_sharing_strategy('file_system')
            except Exception:
                pass

    name = args.out or ('_smoke' if args.smoke else
                        f'{args.dataset}_{args.backbone}_{args.head}'
                        f'{"_nobn" if args.no_bn else ""}'
                        f'{"_noskip" if args.no_skip else ""}_s{args.seed}')
    out = os.path.join(args.runs_dir, name); os.makedirs(out, exist_ok=True)

    # A finished run has no resume checkpoint left, so the configuration guard
    # below has nothing to compare against. A reused --out name is therefore
    # refused instead of overwriting finished results. The install check is exempt.
    if args.smoke:
        args.restart = True
    done = os.path.exists(os.path.join(out, 'DONE'))
    kept = os.path.exists(os.path.join(out, 'ckpt.pt'))
    if done and not kept and not args.restart:
        print(f'{out} already holds a finished run. Nothing was changed.\n'
              f'Use a different --out name, or --restart to overwrite it.')
        return
    # done and kept: finished with --keep-ckpt, so an extension is allowed. Nothing
    # in the folder is touched until the configuration guard below has passed.
    if args.restart:
        for marker in ('DONE', 'DIVERGED', 'ckpt.pt'):
            if os.path.exists(os.path.join(out, marker)):
                os.remove(os.path.join(out, marker))

    train_ld, test_ld, K = get_data(args.dataset, args.data_dir, args.batch,
                                    args.workers, dev, args.fake_data)
    net = make_backbone(args.backbone, not args.no_bn, not args.no_skip).to(dev)
    head = make_head(args.head, net.feat_dim, K, args.eps,
                     args.tau_init, not args.fix_tau, args.tau_floor).to(dev)
    params = list(net.parameters()) + list(head.parameters())
    wd_eta = args.wd if args.wd_eta is None else args.wd_eta
    args.wd_eta = wd_eta                        # recorded in config.json and final.pt
    eta = [p for n, p in head.named_parameters() if n == 'eta']
    rest = [p for n, p in head.named_parameters() if n != 'eta']
    opt = torch.optim.SGD([{'params': list(net.parameters()) + rest},
                           {'params': eta, 'weight_decay': wd_eta}],
                          lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # a CPU mirror of the head, used only for the curvature diagnostic
    head_cpu = make_head(args.head, net.feat_dim, K, args.eps,
                         args.tau_init, not args.fix_tau, args.tau_floor).float()

    print(f'device {dev} | {args.backbone} + {args.head} head | {args.dataset} '
          f'| {sum(p.numel() for p in params)/1e6:.1f}M params | workers {args.workers}'
          f' | -> {out}')

    # ---- resume from the last completed epoch, if there is one -------------
    ck_path = os.path.join(out, 'ckpt.pt')
    start_ep = 0
    GUARD = ['dataset', 'backbone', 'head', 'no_bn', 'no_skip', 'eps', 'tau_init', 'tau_floor',
             'fix_tau', 'tau_freeze', 'fake_data', 'seed', 'lr', 'batch', 'wd_eta']
    if os.path.exists(ck_path) and not args.restart:
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        prev = ck.get('args', {})
        bad = [k for k in GUARD if k in prev and prev[k] != getattr(args, k)]
        if bad:
            print(f'\nERROR: {ck_path} was written by a different configuration.')
            for k in bad:
                print(f'  {k}: checkpoint has {prev[k]!r}, you asked for {getattr(args, k)!r}')
            print('\nResuming would silently continue the OTHER experiment. Either use a'
                  '\nnew --out name, or pass --restart to overwrite this folder.\n')
            raise SystemExit(1)
        net.load_state_dict(ck['net']); head.load_state_dict(ck['head'])
        opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        # The scheduler state carries its own T_max. If --epochs changed, the
        # restored T_max would override the new one and the learning rate would
        # cycle instead of decaying, so rebuild it from args.
        sched.T_max = args.epochs
        start_ep = ck['epoch'] + 1
        print(f'  resuming from epoch {start_ep} (schedule T_max reset to {args.epochs})')

    # the guard has passed (or there was nothing to resume): only now record the
    # configuration and clear a finished marker, so a refused attempt leaves the
    # folder exactly as it was
    with open(os.path.join(out, 'config.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=2)
    if os.path.exists(os.path.join(out, 'DONE')):
        os.remove(os.path.join(out, 'DONE'))

    # NOTE: the 'lr' column is read after sched.step(), so it holds the rate of
    # the NEXT epoch. Epoch e itself ran at lr0 * (1 + cos(pi e / epochs)) / 2.
    fields = ['epoch', 'train_loss', 'train_acc', 'test_acc', 'lr',
              'feat_norm', 'head_curv', 'L_g', 'beta_g', 'kappa_g', 'tau_mean', 'sec']
    if args.head == 'proto':
        fields.append('aux')                    # (max - min) / mean of the apertures
    log_path = os.path.join(out, 'log.csv')
    new_log = start_ep == 0 or not os.path.exists(log_path)
    fh = open(log_path, 'w' if new_log else 'a', newline='')
    wr = csv.DictWriter(fh, fieldnames=fields)
    if new_log:
        wr.writeheader()

    if args.head == 'conic' and args.tau_freeze > 0:
        head.eta.requires_grad_(False)

    for ep in range(start_ep, args.epochs):
        if args.head == 'conic' and args.tau_freeze > 0 and ep == args.tau_freeze:
            head.eta.requires_grad_(True)
            print(f'  tau released at epoch {ep}')
        t0 = time.time(); tot = corr = 0; loss_sum = 0.0
        fnorm = hcurv = 0.0; nb = nd = ndh = 0; diverged = False
        for x, y in train_ld:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            z = net(x)
            logits = head(z)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward(); opt.step()

            loss_sum += loss.item() * y.numel()
            corr += (logits.argmax(1) == y).sum().item(); tot += y.numel()
            if nb % 20 == 0:                       # cheap diagnostic
                with torch.no_grad():
                    fnorm += z.norm(dim=1).mean().item()
                    nd += 1
            if (args.head == 'conic' and args.diag_every > 0
                    and nb % args.diag_every == 0):
              try:
                with torch.no_grad():
                    # on the CPU and on a subsample: the QR and eigenvalue calls
                    # have no MPS kernel
                    m = min(args.diag_samples, z.shape[0])
                    hcurv += head_cpu.head_curvature(
                        z[:m].detach().cpu().float(), y[:m].cpu()).max().item()
                    ndh += 1
              except Exception:
                pass          # eigendecomposition can fail on a diverging run
            nb += 1
        # Stop a run as soon as its loss stops being finite.
        if diverged or not math.isfinite(loss_sum / max(tot, 1)):
            # Record the diverging epoch, so that the run appears in the summaries.
            wr.writerow({'epoch': ep, 'train_loss': float('nan'),
                         'train_acc': round(corr / max(tot, 1), 4), 'test_acc': 0.0,
                         'lr': round(sched.get_last_lr()[0], 6),
                         'feat_norm': '', 'head_curv': '', 'L_g': '', 'beta_g': '',
                         'kappa_g': '', 'tau_mean': '',
                         'sec': round(time.time() - t0, 1)})
            fh.flush(); fh.close()
            open(os.path.join(out, 'DIVERGED'), 'w').write(f'epoch {ep}\n')
            open(os.path.join(out, 'DONE'), 'w').write('diverged\n')
            print(f'  epoch {ep:3d}  loss is not finite -- diverged, stopping')
            return

        sched.step()
        if args.head == 'conic' and args.diag_every > 0:
            head_cpu.load_state_dict({k: v.detach().cpu().float()
                                      for k, v in head.state_dict().items()})
        acc = evaluate(net, head, test_ld, dev)
        b = head.budget() if args.head in ('conic', 'proto') else {}
        row = {
            'epoch': ep, 'train_loss': round(loss_sum / tot, 5),
            'train_acc': round(corr / tot, 4), 'test_acc': round(acc, 4),
            'lr': round(sched.get_last_lr()[0], 6),
            'feat_norm': round(fnorm / max(nd, 1), 3),
            'head_curv': round(hcurv / max(ndh, 1), 4) if ndh else '',
            'L_g': round(float(b['L_g_per_class'].max()), 4) if b else '',
            'beta_g': round(b['beta_g'], 4) if b else '',
            'kappa_g': round(float(b['kappa_g_per_class'].min()), 4) if b else '',
            'tau_mean': round(float(b['tau'].mean()), 4) if 'tau' in b else '',
            'sec': round(time.time() - t0, 1)}
        if args.head == 'proto':
            row['aux'] = round(b['alpha_spread'], 5)
        wr.writerow(row)
        fh.flush()
        torch.save({'net': net.state_dict(), 'head': head.state_dict(),
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': ep, 'args': vars(args)}, ck_path)
        print(f'  epoch {ep:3d}  loss {loss_sum/tot:.4f}  train {corr/tot:.3f}  '
              f'test {acc:.3f}  {time.time()-t0:.1f}s')

    fh.close()
    if not args.no_final:
        torch.save({'net': net.state_dict(), 'head': head.state_dict(),
                    'args': vars(args)}, os.path.join(out, 'final.pt'))
    if not args.keep_ckpt and os.path.exists(ck_path):
        os.remove(ck_path)          # final.pt has the weights; ckpt.pt also
                                    # carries optimiser state and is only for resuming
    open(os.path.join(out, 'DONE'), 'w').write('completed\n')   # lets a sweep skip it
    print(f'done. log: {out}/log.csv')


if __name__ == '__main__':
    main()
