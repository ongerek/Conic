"""
summarize.py -- read every runs/*/log.csv and print one table, or write one CSV.

    python3 src/summarize.py                      all runs, one line each
    python3 src/summarize.py --prefix S_ --group  mean and sd over seeds
    python3 src/summarize.py --dump results/logs.csv
                                                  every epoch of every run in one file
                                                  (read by tables.py and tau_replay.py)

Columns of the table: epochs done, best and final test accuracy, final tau of a
conic head, seconds per epoch, and whether the run diverged.
"""
import argparse, csv, glob, json, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='runs', help='folder that holds the run folders')
ap.add_argument('--prefix', default='')
ap.add_argument('--group', action='store_true',
                help='average over --seed, reporting mean and sd')
ap.add_argument('--dump', metavar='FILE',
                help='write every epoch of every run into one CSV, with the run name '
                     'and its settings on each row')
a = ap.parse_args()

rows = []
for d in sorted(glob.glob(os.path.join(a.runs, a.prefix + '*'))):
    lp = os.path.join(d, 'log.csv')
    if not os.path.exists(lp):
        continue
    L = list(csv.DictReader(open(lp)))
    if not L:
        continue
    cfg = {}
    cp = os.path.join(d, 'config.json')
    if os.path.exists(cp):
        cfg = json.load(open(cp))
    ta = [float(r['test_acc']) for r in L]
    tau = [r['tau_mean'] for r in L if r.get('tau_mean', '') not in ('', None)]
    kap = [r['kappa_g'] for r in L if r.get('kappa_g', '') not in ('', None)]
    rows.append(dict(
        name=os.path.basename(d), n=len(L), cfg=cfg,
        head=cfg.get('head', '?'), bn=('no' if cfg.get('no_bn') else 'yes'),
        lr=cfg.get('lr', float('nan')), eps=cfg.get('eps', float('nan')),
        seed=cfg.get('seed', 0),
        best=max(ta), final=ta[-1],
        tau=float(tau[-1]) if tau else float('nan'),
        kappa=float(kap[-1]) if kap else float('nan'),
        sec=sum(float(r['sec']) for r in L) / len(L),
        div='DIVERGED' if os.path.exists(os.path.join(d, 'DIVERGED')) else ''))

if not rows:
    raise SystemExit('no runs found (looked in %s/)' % a.runs)

if a.dump:
    os.makedirs(os.path.dirname(os.path.abspath(a.dump)), exist_ok=True)
    cols = ['run', 'head', 'bn', 'lr', 'eps', 'seed', 'diverged']
    body = ['epoch', 'train_loss', 'train_acc', 'test_acc', 'lr_now', 'feat_norm',
            'head_curv', 'L_g', 'beta_g', 'kappa_g', 'tau_mean', 'sec', 'aux']
    tail = ['dataset', 'backbone', 'wd_eta', 'tau_floor']
    with open(a.dump, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(cols + body + tail)
        for r in rows:
            cfg = r['cfg']
            pre = [r['name'], r['head'], r['bn'], r['lr'], r['eps'],
                   cfg.get('seed', 0), 1 if r['div'] else 0]
            post = [cfg.get('dataset', 'cifar10'), cfg.get('backbone', 'resnet18'),
                    cfg.get('wd_eta', cfg.get('wd', 5e-4)), cfg.get('tau_floor', 0.0)]
            for e in csv.DictReader(open(os.path.join(a.runs, r['name'], 'log.csv'))):
                w.writerow(pre + [e.get('epoch', ''), e.get('train_loss', ''),
                                  e.get('train_acc', ''), e.get('test_acc', ''),
                                  e.get('lr', ''), e.get('feat_norm', ''),
                                  e.get('head_curv', ''), e.get('L_g', ''),
                                  e.get('beta_g', ''), e.get('kappa_g', ''),
                                  e.get('tau_mean', ''), e.get('sec', ''),
                                  e.get('aux', '')] + post)
    n = sum(1 for _ in open(a.dump)) - 1
    print('wrote %s : %d runs, %d epoch rows, %.0f kB'
          % (a.dump, len(rows), n, os.path.getsize(a.dump) / 1024))
    raise SystemExit

if a.group:
    key = lambda r: (r['head'], r['bn'], r['lr'], r['eps'])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    print('%-8s %4s %7s %6s %3s   %-16s %-16s %s' %
          ('head', 'BN', 'lr', 'eps', 'n', 'best test', 'final test', 'final tau'))
    for k, g in sorted(groups.items()):
        f = lambda c: (st.mean([x[c] for x in g]),
                       st.stdev([x[c] for x in g]) if len(g) > 1 else 0.0)
        b, bs = f('best'); fi, fs = f('final'); t, ts = f('tau')
        print('%-8s %4s %7.4g %6.3g %3d   %.4f +- %.4f  %.4f +- %.4f  %s' %
              (k[0], k[1], k[2], k[3], len(g), b, bs, fi, fs,
               ('%.3f +- %.3f' % (t, ts)) if t == t else '-'))
else:
    print('%-28s %4s %4s %8s %8s %8s %8s %7s %s' %
          ('run', 'ep', 'BN', 'lr', 'best', 'final', 'tau', 's/ep', ''))
    for r in rows:
        print('%-28s %4d %4s %8.4g %8.4f %8.4f %8s %7.0f %s' %
              (r['name'], r['n'], r['bn'], r['lr'], r['best'], r['final'],
               ('%.3f' % r['tau']) if r['tau'] == r['tau'] else '-',
               r['sec'], r['div']))
