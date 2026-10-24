#!/usr/bin/env bash
# scripts/figures.sh -- Figs. 1 to 4 of the paper, with MATLAB if it is installed,
# otherwise with GNU Octave, into results/figures/.
#
#   bash scripts/figures.sh
#
# Fig. 1  matlab/cone_figures.m        no data needed
# Fig. 2  matlab/make_fig_head_slice.m results/slices/S_affine_bn_s0_param_head_span{40,150}.csv
# Fig. 3  matlab/make_fig_r56.m        results/slices/R56_{skip,noskip}_param_backbone_train.csv
# Fig. 4  matlab/make_fig_tau.m        results/figures/tau_{traj,runs}.csv (analyze.sh, step numbers)
# and the synthetic control of Section 6, matlab/crease_control.m (printed to
# results/crease_control.txt).
#
# A figure whose input is missing is skipped. QUICK=1 works in results_quick/.
# On a Linux machine without a display, Octave needs xvfb-run and ghostscript:
#   sudo apt install -y octave gnuplot xvfb ghostscript fonts-freefont-otf

set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
RES=results; [ "${QUICK:-0}" = 1 ] && RES=results_quick
OUT="$RES/figures"; mkdir -p "$OUT"
for f in S_affine_bn_s0_param_head_span40.csv S_affine_bn_s0_param_head_span150.csv \
         R56_skip_param_backbone_train.csv R56_noskip_param_backbone_train.csv; do
  [ -f "$RES/slices/$f" ] && cp "$RES/slices/$f" "$OUT/"
done

CMDS="cone_figures;"
[ -f "$OUT/S_affine_bn_s0_param_head_span150.csv" ] && CMDS="$CMDS make_fig_head_slice;"
ls "$OUT"/R56_*_param_backbone_train.csv >/dev/null 2>&1 && CMDS="$CMDS make_fig_r56;"
[ -f "$OUT/tau_traj.csv" ] && [ -f "$OUT/tau_runs.csv" ] && CMDS="$CMDS make_fig_tau;"
CMDS="$CMDS diary('../crease_control.txt'); crease_control(); diary off;"
echo "figures: $CMDS"

if command -v matlab >/dev/null 2>&1; then
  matlab -batch "addpath('$ROOT/matlab'); cd('$ROOT/$OUT'); $CMDS"
elif command -v octave >/dev/null 2>&1; then
  RUN="octave --no-gui --quiet"
  command -v xvfb-run >/dev/null 2>&1 && [ -z "${DISPLAY:-}" ] && RUN="xvfb-run -a $RUN"
  $RUN --eval "addpath('$ROOT/matlab'); cd('$ROOT/$OUT'); \
               set(0, 'defaultfigurevisible', 'off'); $CMDS" 2>&1 \
    | grep -v -i "^warning\|called from\|^ *line [0-9]\|multiplot\|^ *\^"
else
  echo "Neither matlab nor octave was found (see README, Figures)."; exit 1
fi
ls -1 "$OUT"/*.pdf "$OUT"/*.png 2>/dev/null
