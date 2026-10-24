#!/usr/bin/env bash
# scripts/selftest.sh -- checks that the installation and the scripts work.
#
#   bash scripts/selftest.sh         self-tests of the code (about 1 min on a GPU machine,
#                                    3-4 min on a laptop CPU); no data are downloaded
#   bash scripts/selftest.sh full    the same, then the whole pipeline (train, analyze,
#                                    figures) in miniature on random images, in a
#                                    temporary copy of the repository (about 40 min on a
#                                    two-core laptop CPU, much less on a workstation)
#
# The last line is SELFTEST PASSED or SELFTEST FAILED.

set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PY:-}
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=$ROOT/.venv/bin/python; else PY=python3; fi
fi
fail=0
step() { echo; echo "== $1"; }
check() { if "$@"; then echo "   ok"; else echo "   FAILED: $*"; fail=1; fi; }
in_dir() { local d=$1; shift; ( cd "$d" && "$@" ); }

step "environment"
"$PY" - <<'EOF' || fail=1
import sys, torch, torchvision, numpy, scipy
print('   python %s | torch %s | torchvision %s | numpy %s | scipy %s'
      % (sys.version.split()[0], torch.__version__, torchvision.__version__, numpy.__version__, scipy.__version__))
if torch.cuda.is_available():
    print('   CUDA device: %s' % torch.cuda.get_device_name(0))
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    print('   Apple MPS device')
else:
    print('   no GPU found: the self-tests run on the CPU')
EOF

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

step "conic head: Hessian, curvature, budget, conic type (src/conic_head.py)"
check "$PY" src/conic_head.py
step "training loop: 2 epochs on random images (src/train.py --smoke)"
check "$PY" src/train.py --smoke --runs-dir "$TMP/runs"
step "probe: margin solver and the whole probe on random checkpoints (src/probe.py --selftest)"
check in_dir "$TMP" "$PY" "$ROOT/src/probe.py" --selftest
step "planes and refined lines on a random ResNet-56 (src/crease_lines.py --selftest)"
check in_dir "$TMP" "$PY" "$ROOT/src/crease_lines.py" --selftest

if [ "${1:-}" = "full" ]; then
  step "whole pipeline in miniature (QUICK=1), in $TMP/repo"
  mkdir -p "$TMP/repo" && cp -r src scripts matlab "$TMP/repo/"
  [ -d .venv ] && ln -s "$ROOT/.venv" "$TMP/repo/.venv"
  ( cd "$TMP/repo" && QUICK=1 PY="$PY" LANES=2 bash scripts/train.sh all > train.out 2>&1 ) \
    && tail -n 12 "$TMP/repo/train.out" | sed 's/^/   /'
  ( cd "$TMP/repo" && QUICK=1 PY="$PY" PROBE_JOBS=2 bash scripts/analyze.sh > analyze.out 2>&1 ) \
    && sed 's/^/   /' "$TMP/repo/analyze.out"
  for f in logs.csv probe/S/probe_runs.csv probe/R56/probe_finallayer.csv \
           slices/S_affine_bn_s0_param_head_span150.csv slices/R56_skip_param_backbone_train.csv \
           slices/R56_noskip_zoom_wall.csv crease/crease_lines.csv tables.txt final_layer.txt \
           wrinkles.txt kink_test.txt crease_stats.txt toy_kinks.txt geometry.txt figures/tau_traj.csv; do
    check test -s "$TMP/repo/results_quick/$f"
  done
  for f in "$TMP"/repo/results_quick/*.txt "$TMP"/repo/results_quick/probe/*/probe.log; do
    if grep -q "Traceback\|Error\|FAILED" "$f"; then echo "   FAILED: $f"; sed 's/^/     /' "$f" | tail -5; fail=1; fi
  done
  if command -v matlab >/dev/null 2>&1 || command -v octave >/dev/null 2>&1; then
    ( cd "$TMP/repo" && QUICK=1 bash scripts/figures.sh > figures.out 2>&1 )
    for f in cone_envelope.pdf cone_cells.pdf fig_head_slice.pdf fig_r56_skip.pdf tau_trajectories.pdf; do
      check test -s "$TMP/repo/results_quick/figures/$f"
    done
    [ "$fail" = 0 ] || tail -20 "$TMP/repo/figures.out" | sed 's/^/     /'
  else
    echo "   (no MATLAB or Octave: figures not tested)"
  fi
fi

echo
if [ "$fail" = 0 ]; then echo "SELFTEST PASSED"; else echo "SELFTEST FAILED"; exit 1; fi
