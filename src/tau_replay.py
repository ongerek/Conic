"""
tau_replay.py -- the trajectories of tau and the paths of weight decay alone (Fig. 4,
Table 3), from the epoch log written by summarize.py --dump.

    python3 src/tau_replay.py results/logs.csv --out results/figures

Writes
    tau_traj.csv   tau after every epoch of the 100-epoch conic runs of Table 3
                   (S_conic_*, CTRL_conic_*, LR8_conic_*), with the decay-only path
    tau_runs.csv   final tau of the 20-epoch learning-rate sweep (LR_*), with the
                   decay-only value after 20 epochs
which matlab/make_fig_tau.m reads.

Decay-only path. Weight decay alone drives eta towards 0 and tau = softplus(eta)
towards ln 2. The path is replayed with SGD with momentum 0.9 and weight decay 5e-4
on eta alone, without the loss, starting from tau = 1.5, with 390 steps per epoch
(50000 images, batch 128, last batch dropped) and with the rate that each epoch
used, lr0 (1 + cos(pi e / E)) / 2 for e = 0 .. E-1. The 'lr' column of log.csv is
read after the scheduler step and therefore holds the rate of the next epoch; the
replay does not use it.
"""
import argparse, csv, math, os
from collections import defaultdict

STEPS, WD, MOM, TAU0 = 390, 5e-4, 0.9, 1.5


def decay_path(lr0, epochs, steps=STEPS, wd=WD, mom=MOM, tau0=TAU0):
    """tau after each epoch when only weight decay acts on eta."""
    eta, buf, out = math.log(math.expm1(tau0)), 0.0, []
    for e in range(epochs):
        lr = 0.5 * lr0 * (1 + math.cos(math.pi * e / epochs))
        for _ in range(steps):
            buf = mom * buf + wd * eta
            eta -= lr * buf
        out.append(math.log1p(math.exp(eta)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='?', default='results/logs.csv')
    ap.add_argument('--out', default='results/figures')
    a = ap.parse_args()
    rows = defaultdict(list)
    for r in csv.DictReader(open(a.logs)):
        rows[r['run']].append(r)
    os.makedirs(a.out, exist_ok=True)

    traj_runs = sorted(n for n in rows if n.startswith(('S_conic_', 'CTRL_conic_', 'LR8_conic_')))
    with open(os.path.join(a.out, 'tau_traj.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'bn', 'lr0', 'epoch', 'lr', 'tau', 'tau_wd_only'])
        for n in traj_runs:
            R = rows[n]; lr0 = float(R[0]['lr']); E = len(R)
            path = decay_path(lr0, E)
            for r in R:
                e = int(r['epoch'])
                w.writerow([n, r['bn'], lr0, e,
                            '%.6g' % (0.5 * lr0 * (1 + math.cos(math.pi * e / E))),
                            r['tau_mean'], '%.4f' % path[e]])

    sweep = sorted(n for n in rows if n.startswith('LR_'))
    with open(os.path.join(a.out, 'tau_runs.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'head', 'bn', 'lr', 'epochs_done', 'test_acc_final', 'tau_final',
                    'tau_wd_only'])
        for n in sweep:
            R = rows[n]; lr0 = float(R[0]['lr']); last = R[-1]
            w.writerow([n, R[0]['head'], R[0]['bn'], lr0, len(R), last['test_acc'],
                        last['tau_mean'], '%.4f' % decay_path(lr0, 20)[-1]])
    print('wrote %s/tau_traj.csv (%d runs) and %s/tau_runs.csv (%d runs)'
          % (a.out, len(traj_runs), a.out, len(sweep)))
    for lr0 in (0.1, 0.08, 0.01):
        print('  decay only, 100 epochs from tau = 1.5 at lr %.2f: tau = %.3f'
              % (lr0, decay_path(lr0, 100)[-1]))


if __name__ == '__main__':
    main()
