"""
tables.py -- Tables 2 to 6 of the paper and the counts quoted with them, from the
epoch log (summarize.py --dump) and the probe results (probe.py).

    python3 src/tables.py --logs results/logs.csv --probe results/probe --out results/tables.txt

--logs and --probe accept several files or folders, so that results gathered on
different machines can be read together; a run that appears twice is taken from
the file named last. A table whose runs are missing is skipped with a note.

p-values are exact two-sided permutation tests of the difference of the means over
all C(10,5) = 252 relabellings of two groups of five (the smallest attainable value
is 2/252 = 0.0079). Variance ratios are compared with the F distribution (two-sided).
The bootstrap uses 10000 resamples with a fixed seed.
"""
from __future__ import annotations
import argparse, csv, glob, itertools, math, os, statistics as st, sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_replay import decay_path

OUT = []


def say(s=''):
    OUT.append(s); print(s)


# ------------------------------------------------------------------ reading
def read_logs(files):
    runs = OrderedDict()
    for f in files:
        cur = OrderedDict()
        for r in csv.DictReader(open(f)):
            cur.setdefault(r['run'], []).append(r)
        runs.update(cur)
    return runs


def read_probe(paths, name):
    rows = OrderedDict()
    for p in paths:
        f = os.path.join(p, name) if os.path.isdir(p) else (
            p if os.path.basename(p) == name else os.path.join(os.path.dirname(p), name))
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            key = (r['run'], r.get('k', ''), r.get('split', ''))
            rows[key] = r
    return list(rows.values())


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


# ------------------------------------------------------------------ statistics
def perm_p(a, b):
    """Exact two-sided permutation test of the difference of the means."""
    pool = list(a) + list(b); n = len(a)
    obs = abs(st.mean(a) - st.mean(b)); hit = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        s = set(idx)
        x = [pool[i] for i in s]; y = [pool[i] for i in range(len(pool)) if i not in s]
        hit += abs(st.mean(x) - st.mean(y)) >= obs - 1e-12; tot += 1
    return hit / tot


def betainc(a, b, x):
    """Regularized incomplete beta function (continued fraction, Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(400):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + num * d; d = 1 / d if abs(d) > 1e-300 else 1e300
        c = 1 + num / c if abs(c) > 1e-300 else 1e300
        f *= c * d
        if abs(1 - c * d) < 1e-14:
            break
    return math.exp(lbt) * (f - 1) / a


def f_test(a, b):
    """Variance ratio var(a)/var(b) and its two-sided p-value."""
    va, vb = st.variance(a), st.variance(b)
    F = va / vb; d1, d2 = len(a) - 1, len(b) - 1
    cdf = betainc(d1 / 2, d2 / 2, d1 * F / (d1 * F + d2))
    return F, min(1.0, 2 * min(cdf, 1 - cdf))


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k + 1.0
        return r
    rx, ry = rank(x), rank(y); mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def ms(v, scale=1.0, nd=2):
    v = [x * scale for x in v]
    return ('%.*f +- %.*f' % (nd, st.mean(v), nd, st.stdev(v))) if len(v) > 1 else '%.*f' % (nd, v[0])


# ------------------------------------------------------------------ helpers
def seeds(runs, stem, n=5):
    names = ['%s_s%d' % (stem, s) for s in range(n)]
    return names if all(x in runs for x in names) else None


def final(runs, name, col):
    return fl(runs[name][-1][col])


def note_missing(title, what):
    say('%s: skipped (%s not found)' % (title, what)); say()


# ------------------------------------------------------------------ Table 2
def table2(runs):
    say('Table 2. Final test accuracy (%), mean +- sd over five seeds')
    sets = [('batch norm, lr 0.1', 'S_affine_bn', 'S_conic_bn'),
            ('no batch norm, lr 0.01', 'S_affine_nobn', 'S_conic_nobn'),
            ('no batch norm, lr 0.08', 'LR8_affine_nobn', 'LR8_conic_nobn')]
    say('%-24s %-16s %-16s %14s %8s' % ('setting', 'affine', 'conic', 'conic - affine', 'p'))
    any_ = False
    for lab, a, c in sets:
        A, C = seeds(runs, a), seeds(runs, c)
        if not (A and C):
            say('%-24s (runs %s_s* / %s_s* not found)' % (lab, a, c)); continue
        x = [100 * final(runs, n, 'test_acc') for n in A]; y = [100 * final(runs, n, 'test_acc') for n in C]
        sep = 'every conic run above every affine run' if min(y) > max(x) else ''
        say('%-24s %-16s %-16s %+14.2f %8.4f  %s' % (lab, ms(x), ms(y), st.mean(y) - st.mean(x),
                                                    perm_p(x, y), sep))
        any_ = True
    say()
    return any_


# ------------------------------------------------------------------ Table 3
def table3(runs):
    sets = [('batch norm, lr 0.1', 'S_conic_bn', 0.1), ('batch norm, lr 0.01', 'CTRL_conic_bn_lr001', 0.01),
            ('no batch norm, lr 0.01', 'S_conic_nobn', 0.01), ('no batch norm, lr 0.08', 'LR8_conic_nobn', 0.08)]
    have = {stem: seeds(runs, stem) for _, stem, _ in sets}
    say('Table 3. Final tau of the conic head over five seeds, with the decay-only value')
    say('%-24s %-18s %-14s %10s' % ('setting', 'tau, mean +- sd', 'range', 'decay only'))
    T = {}
    for lab, stem, lr in sets:
        if not have[stem]:
            say('%-24s (runs %s_s* not found)' % (lab, stem)); continue
        t = [final(runs, n, 'tau_mean') for n in have[stem]]; T[stem] = (t, decay_path(lr, 100)[-1])
        say('%-24s %-18s %.3f-%.3f    %10.3f' % (lab, ms(t, nd=3), min(t), max(t), T[stem][1]))
    if 'CTRL_conic_bn_lr001' in T and 'S_conic_nobn' in T:
        a, da = T['CTRL_conic_bn_lr001']; b, db = T['S_conic_nobn']
        say('  matched rate 0.01: with BN %.2f to %.2f below the path, without BN %.2f to %.2f above it'
            % (min(da - x for x in a), max(da - x for x in a), min(x - db for x in b), max(x - db for x in b)))
        F, pF = f_test(b, a) if st.variance(b) > st.variance(a) else f_test(a, b)
        say('  location p = %.4f; variance ratio %.2f, p = %.2f' % (perm_p(a, b), F, pF))
    if 'S_conic_bn' in T and 'LR8_conic_nobn' in T:
        a, _ = T['S_conic_bn']; b, _ = T['LR8_conic_nobn']
        F, pF = f_test(a, b)
        say('  larger rates: location p = %.3f; variance ratio %.1f, p = %.3f' % (perm_p(a, b), F, pF))
    for stem in ('S_conic_bn', 'LR8_conic_nobn'):
        if have[stem]:
            lo = [min(fl(r['tau_mean']) for r in runs[n]) for n in have[stem]]
            ep = [min(range(len(runs[n])), key=lambda i: fl(runs[n][i]['tau_mean'])) for n in have[stem]]
            say('  %s: lowest tau %.2f-%.2f, reached at epochs %s' % (stem, min(lo), max(lo), ep))
    say()


# ------------------------------------------------------------------ Table 4
def table4(runs):
    if not any(n.startswith('LR_') for n in runs):
        return note_missing('Table 4', 'LR_* runs')
    say('Table 4. Test accuracy after 20 epochs against the learning rate (D: diverged, ---: not run)')

    def cell(n):
        if n not in runs:
            return '---', '---'
        R = runs[n]
        if R[0].get('diverged', '0') in ('1', 'True') or not math.isfinite(fl(R[-1]['train_loss'])):
            return 'D', '---'
        t, acc = fl(R[-1]['tau_mean']), fl(R[-1]['test_acc'])
        chance = acc <= 0.11                      # tau is not reported for a run at chance
        return '%.4f' % acc, ('%.3f' % t if math.isfinite(t) and not chance else '---')
    for bn, lrs in (('bn', [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.2, 1.6, 2.4]),
                    ('nobn', [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])):
        say('  %s' % ('batch norm' if bn == 'bn' else 'no batch norm'))
        say('  %6s %8s %8s %8s' % ('lr', 'affine', 'conic', 'tau'))
        for lr in lrs:
            a, _ = cell('LR_affine_%s_%g' % (bn, lr)); c, t = cell('LR_conic_%s_%g' % (bn, lr))
            if a == c == '---':
                continue
            say('  %6g %8s %8s %8s' % (lr, a, c, t))
    say()


# ------------------------------------------------------------------ Table 5
def table5(runs):
    rows = [('affine, batch norm', 'BASE_affine'), ('conic eps 0.1, tau free', 'cifar10_resnet18_conic_s0'),
            ('conic eps 0.5, tau >= 1', 'A_floor1_eps05'), ('conic eps 0.5, tau frozen 5 ep', 'F_taufreeze5'),
            ('affine, no batch norm, lr 0.01', 'E_affine_nobn'),
            ('conic eps 0.5, no batch norm, lr 0.01', 'C_nobn_freetau_eps05'),
            ('conic eps 0.5, CIFAR-100, batch norm', 'D_cifar100_freetau')]
    if not any(n in runs for _, n in rows):
        return note_missing('Table 5', 'single-seed runs')
    say('Table 5. Single-seed runs')
    say('%-40s %-28s %8s %8s' % ('configuration', 'run', 'accuracy', 'tau'))
    for lab, n in rows:
        if n in runs:
            t = final(runs, n, 'tau_mean')
            say('%-40s %-28s %8.4f %8s' % (lab, n, final(runs, n, 'test_acc'),
                                           '%.3f' % t if math.isfinite(t) else '---'))
        else:
            say('%-40s %-28s %8s' % (lab, n, 'not found'))
    say()


# ------------------------------------------------------------------ probe-based
def by_run(probe):
    return OrderedDict((r['run'], r) for r in probe)


def group(P, stem, n=5):
    names = ['%s_s%d' % (stem, s) for s in range(n)]
    return [P[x] for x in names] if all(x in P for x in names) else None


def section72(P):
    sets = [('batch norm, lr 0.1', 'S_affine_bn', 'S_conic_bn'),
            ('no batch norm, lr 0.01', 'S_affine_nobn', 'S_conic_nobn'),
            ('no batch norm, lr 0.08', 'LR8_affine_nobn', 'LR8_conic_nobn')]
    if not any(group(P, c) for _, _, c in sets):
        return note_missing('Section 7.2', 'probe rows of S_* / LR8_*')
    say('Section 7.2. Affine against conic head, five seeds each (mean +- sd; p: permutation test)')
    for lab, a, c in sets:
        A, C = group(P, a), group(P, c)
        if not (A and C):
            continue
        say('  %s' % lab)
        x = [fl(r['fnorm_test']) for r in A]; y = [fl(r['fnorm_test']) for r in C]
        say('    %-34s affine %-16s conic %-16s reduction %4.1f%%  p = %.4f%s' % (
            'mean norm of the test features', ms(x), ms(y), 100 * (1 - st.mean(y) / st.mean(x)),
            perm_p(x, y), '  (every run separated)' if max(y) < min(x) or min(y) > max(x) else ''))
        for key, nm, nd in (('gap', 'logit gap (top 1 - top 2)', 2), ('gain', 'gain |grad_z gap|', 3),
                            ('norm_x_gain', 'feature norm x gain', 2), ('ece', 'ECE (15 bins)', 4),
                            ('auroc_msp_cifar100', 'AUROC max softmax vs CIFAR-100', 3),
                            ('auroc_fnorm_cifar100', 'AUROC feature norm vs CIFAR-100', 3)):
            if key not in A[0]:
                continue
            x = [fl(r[key]) for r in A]; y = [fl(r[key]) for r in C]
            sep = '  (every run separated)' if max(y) < min(x) or min(y) > max(x) else ''
            say('    %-34s affine %-16s conic %-16s p = %.4f%s' % (nm, ms(x, nd=nd), ms(y, nd=nd),
                                                                  perm_p(x, y), sep))
        if 'fnorm_cifar100' in A[0]:
            ra = st.mean(fl(r['fnorm_cifar100']) / fl(r['fnorm_test']) for r in A)
            rc = st.mean(fl(r['fnorm_cifar100']) / fl(r['fnorm_test']) for r in C)
            say('    %-34s affine %.0f%%, conic %.0f%%' % ('outside norm / inside norm', 100 * ra, 100 * rc))
    say()


def forces(P, classes):
    runs = [r for r in P if r.startswith(('S_conic_', 'CTRL_conic_', 'LR8_conic_'))]
    cc = [c for c in classes if c['run'] in runs and 'dLdeta_train' in c]
    if not cc:
        return note_missing('Section 7.4 (forces on eta)', 'probe classes of S_conic_* / CTRL_* / LR8_conic_*')
    say('Section 7.4. Forces on eta_k at the final checkpoint, five-seed conic runs (%d runs)' % len(runs))
    for split in ('train', 'train_aug'):
        down, up, mean_ratio = [], [], []
        for run in runs:
            rc = [c for c in cc if c['run'] == run]
            g = [fl(c['dLdeta_' + split]) for c in rc]; w = [fl(c['wd_force_' + split]) for c in rc]
            down.append(sum(1 for x in g if x > 0))
            up += [-x / v for x, v in zip(g, w) if x < 0]
            mean_ratio.append(abs(st.mean(g)) / st.mean(w))
        say('  %-9s classes pushed down per run %d-%d; push of the others / weight-decay pull %.2f-%.1f '
            '(median %.2f, %d%% within a factor 1.25); |mean push| / mean pull %.1f%%-%.0f%%'
            % (split, min(down), max(down), min(up), max(up), st.median(up),
               round(100 * sum(1 for r in up if 0.8 <= r <= 1.25) / len(up)),
               100 * min(mean_ratio), 100 * max(mean_ratio)))
    say()


def table6(P):
    cols = [('batch norm, lr 0.1', {'affine': 'G_affine_bn', 'conic': 'G_conic_bn', 'nwd': 'NWD_conic_bn',
                                    'proto': 'PROTO_bn', 'cosine': 'COS_bn'},
             {'affine': 'S_affine_bn', 'conic': 'S_conic_bn'}),
            ('no batch norm, lr 0.01', {'affine': 'G_affine_nobn', 'conic': 'G_conic_nobn',
                                        'nwd': 'NWD_conic_nobn', 'proto': None, 'cosine': 'COS_nobn'},
             {'affine': 'S_affine_nobn', 'conic': 'S_conic_nobn'}),
            ('no batch norm, lr 0.08', {'affine': 'G_affine_nobn_lr008', 'conic': 'G_conic_nobn_lr008',
                                        'nwd': 'NWD_conic_nobn_lr008', 'proto': 'PROTO_nobn_lr008',
                                        'cosine': 'COS_nobn_lr008'},
             {'affine': 'LR8_affine_nobn', 'conic': 'LR8_conic_nobn'})]
    if not any(group(P, s) for _, m, _ in cols for s in m.values() if s):
        return note_missing('Table 6', 'probe rows of G_* / NWD_* / PROTO_* / COS_*')
    say('Table 6. Accuracy (%), mean test feature norm and AUROC of the feature norm (CIFAR-10 vs')
    say('CIFAR-100 test images), means over five seeds')
    labels = OrderedDict([('affine', 'affine'), ('conic', 'conic'), ('nwd', 'conic, eta without decay'),
                          ('proto', 'prototype'), ('cosine', 'cosine')])
    say('%-26s' % 'head' + ''.join('| %-27s' % c[0] for c in cols))
    fallback = False
    for key, lab in labels.items():
        line = '%-26s' % lab
        for _, m, fb in cols:
            stem = m[key]
            G = group(P, stem) if stem else None
            if G is None and key in fb:
                G = group(P, fb[key]); fallback |= G is not None
            if G is None:
                line += '| %-27s' % '---'; continue
            acc = st.mean(100 * fl(r['acc']) for r in G); fn = st.mean(fl(r['fnorm_test']) for r in G)
            au = st.mean(fl(r.get('auroc_fnorm_cifar100', 'nan')) for r in G)
            line += '| %6.2f %7.3g %6.3f      ' % (acc, fn, au)
        say(line)
    if fallback:
        say('  (affine and conic rows without G_* runs are taken from S_* / LR8_*)')
    # comparisons quoted in Section 7.7
    pairs = [('NWD_conic_bn', 'G_conic_bn'), ('NWD_conic_bn_lr001', 'G_conic_bn_lr001'),
             ('NWD_conic_nobn', 'G_conic_nobn'), ('NWD_conic_nobn_lr008', 'G_conic_nobn_lr008'),
             ('PROTO_bn', 'G_affine_bn'), ('PROTO_nobn_lr008', 'G_affine_nobn_lr008'),
             ('COS_bn', 'G_affine_bn'), ('COS_nobn', 'G_affine_nobn'), ('COS_nobn_lr008', 'G_affine_nobn_lr008')]
    for x, y in pairs:
        X, Y = group(P, x), group(P, y)
        if X and Y:
            a = [100 * fl(r['acc']) for r in X]; b = [100 * fl(r['acc']) for r in Y]
            say('  accuracy %-22s %6.2f vs %-20s %6.2f  p = %.3f' % (x, st.mean(a), y, st.mean(b), perm_p(a, b)))
    for stem in ('NWD_conic_bn', 'NWD_conic_bn_lr001', 'NWD_conic_nobn', 'NWD_conic_nobn_lr008'):
        X = group(P, stem)
        if X:
            t = [fl(r['tau']) for r in X]
            say('  tau %-22s %s  (range %.2f-%.2f)' % (stem, ms(t, nd=3), min(t), max(t)))
    for a, b in (('NWD_conic_bn', 'G_conic_bn'), ('NWD_conic_nobn', 'G_conic_nobn'),
                 ('NWD_conic_nobn_lr008', 'G_conic_nobn_lr008')):
        X, Y = group(P, a), group(P, b)
        if X and Y:
            t = [fl(r['tau']) for r in X + Y]; f = [fl(r['fnorm_test']) for r in X + Y]
            say('  Spearman(tau, feature norm) over %s and %s: %+.2f' % (a, b, spearman(t, f)))
    say()


def cifar100(P, runs):
    A, C = group(P, 'C100_affine_bn'), group(P, 'C100_conic_bn')
    if not (A and C):
        return note_missing('CIFAR-100 (Section 7.7)', 'probe rows of C100_*')
    say('CIFAR-100, five seeds per head')
    x = [100 * fl(r['acc']) for r in A]; y = [100 * fl(r['acc']) for r in C]
    say('  accuracy: affine %s, conic %s, p = %.3f' % (ms(x), ms(y), perm_p(x, y)))
    x = [fl(r['fnorm_test']) for r in A]; y = [fl(r['fnorm_test']) for r in C]
    say('  feature norm: affine %.2f, conic %.2f (%.0f%% smaller), p = %.3f'
        % (st.mean(x), st.mean(y), 100 * (1 - st.mean(y) / st.mean(x)), perm_p(x, y)))
    say('  tau: %s' % ms([fl(r['tau']) for r in C], nd=3))
    say()


def cells(P, classes):
    conic = [r for r in P.values() if r['head'] == 'conic']
    if not conic:
        return note_missing('Cells (Section 7.1)', 'conic heads in the probe')
    say('Cells of the trained conic heads (Proposition 7, Corollary 8, Sections 7.1 and 7.7)')
    say('  family = first part of the run name; delta = distance from w_k to the convex hull of')
    say('  the other axes; excess = alpha_k - min_j alpha_j, both relative to |w_k|')
    K = lambda r: 100 if r.get('dataset') == 'cifar100' else 10
    fam = OrderedDict()
    for r in conic:
        fam.setdefault(r['run'].split('_')[0], []).append(r)
    fam['all'] = conic
    byrun = {r['run']: r for r in conic}
    say('  %-6s %5s %6s %9s %7s %10s %15s %8s %10s' % ('family', 'heads', 'cells', 'unbounded',
                                                       'bounded', 'escape min', 'delta range', 'median',
                                                       'excess max'))
    for f, rs in fam.items():
        names = {r['run'] for r in rs}
        tot = sum(K(r) for r in rs); op = sum(int(fl(r['n_open'])) for r in rs)
        cl = sum(int(fl(r['n_closed'])) for r in rs)
        for ds in ('cifar10', 'cifar100'):
            cc = [c for c in classes if c['head'] == 'conic' and c['run'] in names
                  and byrun[c['run']].get('dataset', 'cifar10') == ds and fl(c.get('w_norm_k')) > 0]
            if not cc:
                continue
            d = sorted(fl(c['dist_conv']) / fl(c['w_norm_k']) for c in cc)
            exc = max(-fl(c['min_dbeta']) / fl(c['w_norm_k']) for c in cc)
            dec = sum(fl(c['bound']) > 0 for c in cc)
            esc = min(fl(r['escape']) for r in rs if r.get('dataset', 'cifar10') == ds)
            nh = sum(1 for r in rs if r.get('dataset', 'cifar10') == ds)
            nc = sum(K(r) for r in rs if r.get('dataset', 'cifar10') == ds)
            no = sum(int(fl(r['n_open'])) for r in rs if r.get('dataset', 'cifar10') == ds)
            nb = sum(int(fl(r['n_closed'])) for r in rs if r.get('dataset', 'cifar10') == ds)
            say('  %-6s %5d %6d %9d %7d %10.3f %7.3f-%.3f %8.3f %10.3f   %s, Corollary 8 decides %d of %d'
                % (f, nh, nc, no, nb, esc, d[0], d[-1], st.median(d), exc, ds, dec, len(cc)))
    aff = [r for r in P.values() if r['head'] == 'affine' and math.isfinite(fl(r.get('escape')))]
    if aff:
        say('  affine heads %d: escaping share of the correctly classified test features at least %.3f'
            % (len(aff), min(fl(r['escape']) for r in aff)))
    proto = [r for r in P.values() if r['head'] == 'proto']
    if proto:
        say('  prototype heads %d: unbounded cells per head %s, escape %.3f-%.3f, aperture CV %.3f-%.3f'
            % (len(proto), sorted(set(int(fl(r['n_open'])) for r in proto)),
               min(fl(r['escape']) for r in proto), max(fl(r['escape']) for r in proto),
               min(fl(r['alpha_cv']) for r in proto), max(fl(r['alpha_cv']) for r in proto)))
        pc = [c for c in classes if c['head'] == 'proto']
        if pc:
            say('  prototype margins M_k: %.2f to %.2f' % (min(fl(c['M_lo']) for c in pc),
                                                          max(fl(c['M_hi']) for c in pc)))
    say()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs='+', default=['results/logs.csv'])
    ap.add_argument('--probe', nargs='+', default=['results/probe'])
    ap.add_argument('--out', default=None, help='also write the tables to this text file')
    a = ap.parse_args()
    files = [f for f in a.logs if os.path.exists(f)]
    runs = read_logs(files) if files else OrderedDict()
    probe = read_probe(a.probe, 'probe_runs.csv')
    classes = read_probe(a.probe, 'probe_classes.csv')
    say('logs: %s (%d runs) | probe: %d runs' % (', '.join(files) or 'none', len(runs), len(probe)))
    say()
    if runs:
        table2(runs); table3(runs); table4(runs); table5(runs)
    P = by_run(probe)
    if P:
        section72(P); forces(P, classes); table6(P); cifar100(P, runs); cells(P, classes)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, 'w').write('\n'.join(OUT) + '\n')
        print('wrote', a.out)


if __name__ == '__main__':
    main()
