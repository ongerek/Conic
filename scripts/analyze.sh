#!/usr/bin/env bash
# scripts/analyze.sh -- every measurement of the paper on the trained runs, then the numbers.
#
#   bash scripts/analyze.sh                 all steps, in order
#   bash scripts/analyze.sh probe numbers   selected steps
#   nohup bash scripts/analyze.sh > logs/analyze.log 2>&1 &
#
# Steps (each one skips outputs that already exist; FORCE=1 recomputes them):
#   logs      every epoch of every run in one file           results/logs.csv
#   probe     probe.py on every final.pt, one process per run family, PROBE_JOBS
#             families at a time (default 4)                   results/probe/<family>/
#   slices    final-layer slices of Fig. 2, backbone slices of Fig. 3, the 4096-image
#             slices and the six refined windows of Table 1    results/slices/
#   crease    twenty planes and forty refined lines of Section 6   results/crease/
#   geometry  flatness of the learned boundaries (Section 7.8)     results/geometry.txt
#   numbers   tables and numbers of Sections 5-7, the data of Fig. 4, and the toy
#             network of Proposition 11                          results/*.txt
#
# Steps that need runs which are not there are skipped with a note, so the script
# can be used after any subset of scripts/train.sh. QUICK=1 works on runs_quick/ and
# results_quick/ with tiny settings on random images (tests the scripts only).

set -u
cd "$(dirname "$0")/.."
PY=${PY:-}
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
QUICK=${QUICK:-0}
FORCE=${FORCE:-0}
RUNS=runs; RES=results
FAKE=""
if [ "$QUICK" = 1 ]; then RUNS=runs_quick; RES=results_quick; FAKE="--fake-data --device cpu"; fi
mkdir -p "$RES" logs
NCPU=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

STEPS=" ${*:-logs probe slices crease geometry numbers} "
need() { [ "$FORCE" = 1 ] || [ ! -s "$1" ]; }          # recompute when missing or forced
have() { [ -f "$RUNS/$1/final.pt" ]; }
say()  { echo "$(date '+%F %T')  $*"; }

# ------------------------------------------------------------------ logs
if [[ "$STEPS" == *" logs "* ]]; then
  say "logs"
  "$PY" src/summarize.py --runs "$RUNS" --dump "$RES/logs.csv"
fi

# ------------------------------------------------------------------ probe
if [[ "$STEPS" == *" probe "* ]]; then
  JOBS=${PROBE_JOBS:-4}
  THREADS=$(( NCPU / JOBS )); [ "$THREADS" -lt 1 ] && THREADS=1
  FAMS=$(ls -d "$RUNS"/*/ 2>/dev/null | while read -r d; do [ -f "$d/final.pt" ] && basename "$d"; done \
         | sed 's/_.*//' | sort -u)
  say "probe: families $(echo $FAMS | tr '\n' ' ')($JOBS at a time, $THREADS CPU threads each)"
  if [ "$QUICK" != 1 ]; then      # both CIFAR sets are needed (each is the outside data of the other)
    "$PY" -c "
from torchvision import datasets
for name in ('CIFAR10', 'CIFAR100'):
    for train in (True, False):
        getattr(datasets, name)('./data', train=train, download=True)
" || { echo "Could not read or download CIFAR into ./data"; exit 1; }
  fi
  running=0
  for f in $FAMS; do
    out="$RES/probe/$f"
    if ! need "$out/probe_runs.csv"; then say "  probe $f: exists, skipped"; continue; fi
    mkdir -p "$out"
    ( OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS "$PY" src/probe.py "$RUNS/${f}_*/final.pt" \
        --out "$out" $FAKE > "$out/probe.log" 2>&1
      say "  probe $f: $(tail -n 1 "$out/probe.log")" ) &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
  done
  wait
fi

# ------------------------------------------------------------------ slices
slice() {   # slice <run> <csv> <arguments...>
  local run=$1 csv=$2; shift 2
  if ! have "$run"; then say "  $csv: skipped ($RUNS/$run/final.pt not found)"; return; fi
  if ! need "$RES/slices/$csv"; then say "  $csv: exists, skipped"; return; fi
  say "  $csv"
  "$PY" src/loss_surface.py "$RUNS/$run/final.pt" --mode param "$@" $FAKE \
      --out "$RES/slices/$csv" >> logs/slices.log 2>&1 || say "  $csv FAILED, see logs/slices.log"
}
if [[ "$STEPS" == *" slices "* ]]; then
  say "slices"
  mkdir -p "$RES/slices"
  if [ "$QUICK" = 1 ]; then G41=13; G61=13; G21=7; GZ=13; I512=16; I4096=32; IH=64
  else G41=41; G61=61; G21=21; GZ=61; I512=512; I4096=4096; IH=1024; fi
  # Fig. 2: only the affine final layer moves, test images
  slice S_affine_bn_s0 S_affine_bn_s0_param_head_span40.csv  --scope head --span 40  --grid $G41 --images $IH
  slice S_affine_bn_s0 S_affine_bn_s0_param_head_span150.csv --scope head --span 150 --grid $G61 --images $IH
  for net in R56_skip R56_noskip; do
    # Fig. 3: the whole backbone moves, training images without augmentation
    slice $net ${net}_param_backbone_train.csv      --scope backbone --split train --grid $G41 --images $I512
    # Section 6: the same plane with eight times as many images
    slice $net ${net}_param_backbone_train_4096.csv --scope backbone --split train --grid $G21 --images $I4096
  done
  # Table 1: six refined windows on the same planes, grid spacing 0.005
  z() { slice "$1" "$1_zoom_$2.csv" --scope backbone --split train --images $I512 --center="$3" --span 0.15 --grid $GZ; }
  z R56_skip   dimple  0.70,0.70
  z R56_skip   bottom  0,0
  z R56_skip   wall   -0.80,-0.25
  z R56_noskip dimple -0.55,-0.50
  z R56_noskip bottom  0,0
  z R56_noskip wall   -0.30,-0.80
fi

# ------------------------------------------------------------------ crease
if [[ "$STEPS" == *" crease "* ]]; then
  say "crease"
  if have R56_skip || have R56_noskip; then
    if need "$RES/crease/crease_lines.csv"; then
      SMALL=""; [ "$QUICK" = 1 ] && SMALL="--planes 1 --images 16 --chunk 16 --coarse 5 --points 17"
      "$PY" src/crease_lines.py "$RUNS/R56_skip/final.pt" "$RUNS/R56_noskip/final.pt" \
          --out "$RES/crease" $SMALL $FAKE > logs/crease.log 2>&1
      say "  $(tail -n 1 logs/crease.log)"
    else say "  exists, skipped"; fi
  else say "  skipped (no R56_* checkpoints)"; fi
fi

# ------------------------------------------------------------------ geometry
if [[ "$STEPS" == *" geometry "* ]]; then
  say "geometry"
  # the ten conic heads at the rate 0.01 (with and without batch norm); any conic head in QUICK mode
  GEO=("$RUNS/CTRL_conic_bn_lr001_s*/final.pt" "$RUNS/S_conic_nobn_s*/final.pt")
  [ "$QUICK" = 1 ] && GEO=("$RUNS/S_conic_bn_s*/final.pt")
  if ls ${GEO[@]} >/dev/null 2>&1; then
    need "$RES/geometry.txt" && "$PY" src/geometry.py "${GEO[@]}" > "$RES/geometry.txt" 2>&1
    say "  $RES/geometry.txt"
  else say "  skipped (no CTRL_conic_bn_lr001_* or S_conic_nobn_* checkpoints)"; fi
fi

# ------------------------------------------------------------------ numbers
if [[ "$STEPS" == *" numbers "* ]]; then
  say "numbers"
  mkdir -p "$RES/figures"
  if [ -s "$RES/logs.csv" ]; then
    "$PY" src/tau_replay.py "$RES/logs.csv" --out "$RES/figures" > "$RES/tau_replay.txt"
  fi
  "$PY" src/tables.py --logs "$RES/logs.csv" --probe "$RES"/probe/* --out "$RES/tables.txt" > /dev/null
  say "  $RES/tables.txt"
  S="$RES/slices"
  if [ -s "$S/S_affine_bn_s0_param_head_span150.csv" ]; then
    "$PY" src/final_layer_slice.py "$S/S_affine_bn_s0_param_head_span40.csv" \
        "$S/S_affine_bn_s0_param_head_span150.csv" > "$RES/final_layer.txt" 2>&1
    say "  $RES/final_layer.txt"
  fi
  : > "$RES/wrinkles.txt"
  for net in R56_skip R56_noskip; do
    if [ -s "$S/${net}_param_backbone_train_4096.csv" ]; then
      echo "== $net" >> "$RES/wrinkles.txt"
      "$PY" src/wrinkles.py "$S/${net}_param_backbone_train.csv" \
          "$S/${net}_param_backbone_train_4096.csv" >> "$RES/wrinkles.txt" 2>&1
    fi
  done
  [ -s "$RES/wrinkles.txt" ] && say "  $RES/wrinkles.txt" || rm -f "$RES/wrinkles.txt"
  if ls "$S"/R56_*_zoom_*.csv >/dev/null 2>&1; then
    "$PY" src/kink_test.py "$S"/R56_*_zoom_*.csv > "$RES/kink_test.txt" 2>&1
    say "  $RES/kink_test.txt"
  fi
  if [ -s "$RES/crease/crease_lines.csv" ]; then
    "$PY" src/crease_stats.py "$RES/crease" > "$RES/crease_stats.txt" 2>&1
    say "  $RES/crease_stats.txt"
  fi
  if need "$RES/toy_kinks.txt"; then
    "$PY" src/toy_kinks.py > "$RES/toy_kinks.txt" 2>&1
  fi
  say "  $RES/toy_kinks.txt"
fi
say "analyze.sh finished."
