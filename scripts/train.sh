#!/usr/bin/env bash
# scripts/train.sh -- every training run of the paper, on one CUDA GPU, several at a time.
#
#   bash scripts/train.sh main                 one group
#   bash scripts/train.sh main sweep ablation  several groups
#   bash scripts/train.sh all                  all 166 runs
#   nohup bash scripts/train.sh all > logs/train.log 2>&1 &      (unattended; see README)
#
# Groups (run names as in the paper's logs):
#   main       35  S_*, CTRL_*, LR8_*   affine and conic heads, ResNet-18, CIFAR-10,
#                                       5 seeds (Tables 2, 3; Fig. 4a; Sections 5, 7.1-7.2)
#   sweep      32  LR_*                 learning-rate sweep, 20 epochs (Table 4; Fig. 4b)
#   ablation    7  single-seed runs     (Table 5)
#   resnet56    2  R56_skip, R56_noskip (Section 6: Fig. 3, Table 1, twenty planes)
#   nwd        20  NWD_*                eta excluded from weight decay (Table 6, Sec. 7.7)
#   proto      10  PROTO_*              prototype limit of the head, w_k = 0 (Table 6)
#   cosine     15  COS_*                cosine head (Table 6)
#   replicate  35  G_*                  the runs of 'main' again, under new names (Table 6
#                                       uses them; on a single machine they are a replicate)
#   c100       10  C100_*               affine and conic heads on CIFAR-100, 5 seeds (Sec. 7.7)
#
# The runs of a group are divided into LANES lanes that train side by side on the GPU;
# each lane trains its share one run after another. LANES defaults to the number of CPU
# cores divided by 4, at most 4 (data loading needs about four cores per lane).
#
# Environment variables (all optional):
#   LANES=2         runs side by side on the GPU (use 1 on a small GPU or a Mac)
#   WORKERS=4       data-loading processes per run
#   PY=python3      the Python interpreter (default: .venv/bin/python if present)
#   LIST=1          print the training commands of the chosen groups and stop
#   QUICK=1         dry run on random images, 1 epoch, first 2 runs of each group,
#                   in runs_quick/ (tests the scripts, not the science; CPU is fine)
#
# Interrupted? Give the same command again: finished runs are skipped, and an unfinished
# run continues from its last completed epoch.

set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)

WANT=" ${*:-} "
[ "$WANT" = "  " ] && { sed -n '2,/^$/p' "$0"; exit 1; }
[[ "$WANT" == *" all "* ]] && WANT=" main sweep ablation resnet56 nwd proto cosine replicate c100 "
for g in $WANT; do
  case "$g" in main|sweep|ablation|resnet56|nwd|proto|cosine|replicate|c100) ;;
    *) echo "unknown group: $g"; exit 1 ;; esac
done

PY=${PY:-}
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
QUICK=${QUICK:-0}
RUNS=runs; LOGS=logs
[ "$QUICK" = 1 ] && { RUNS=runs_quick; LOGS=logs_quick; }
mkdir -p "$RUNS" "$LOGS"

# one driver at a time per runs folder
if command -v flock >/dev/null 2>&1; then
  exec 9> "$RUNS/.train.lock"
  if ! flock -n 9; then echo "scripts/train.sh is already running for $RUNS/. Nothing was started."; exit 1; fi
fi

if [ "$QUICK" != 1 ]; then
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    if [ "${ALLOW_NO_CUDA:-0}" != 1 ]; then
      echo "PyTorch with CUDA was not found by $PY (see README, Installation)."
      echo "To train on another device anyway (Apple MPS or CPU): ALLOW_NO_CUDA=1 LANES=1 bash scripts/train.sh ..."
      exit 1
    fi
  fi
fi

NCPU=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
LANES=${LANES:-$(( NCPU / 4 ))}
[ "$LANES" -gt 4 ] && LANES=4
[ "$LANES" -lt 1 ] && LANES=1
WORKERS=${WORKERS:-$(( NCPU / LANES - 1 ))}
[ "$WORKERS" -gt 6 ] && WORKERS=6
[ "$WORKERS" -lt 0 ] && WORKERS=0

BASE="--dataset cifar10 --backbone resnet18 --epochs 100 --workers $WORKERS"
EXTRA=""
if [ "$QUICK" = 1 ]; then
  EXTRA="--fake-data --epochs 1 --workers 0 --device cpu"
fi


JOBS=()
add() { JOBS+=("$*"); }
NADD=0
group_start() { NADD=0; }
want() { [[ "$WANT" == *" $1 "* ]]; }
take() { NADD=$((NADD + 1)); [ "$QUICK" != 1 ] || [ "$NADD" -le 2 ]; }
job() { take && add "$@"; }

if want main; then group_start
  for S in 0 1 2 3 4; do
    job --head affine --lr 0.1                         --seed $S --out S_affine_bn_s$S
    job --head conic --eps 0.5 --lr 0.1                --seed $S --out S_conic_bn_s$S
    job --head affine --no-bn --lr 0.01                --seed $S --out S_affine_nobn_s$S
    job --head conic --eps 0.5 --no-bn --lr 0.01       --seed $S --out S_conic_nobn_s$S
    job --head conic --eps 0.5 --lr 0.01               --seed $S --out CTRL_conic_bn_lr001_s$S
    job --head affine --no-bn --lr 0.08                --seed $S --out LR8_affine_nobn_s$S
    job --head conic --eps 0.5 --no-bn --lr 0.08       --seed $S --out LR8_conic_nobn_s$S
  done
fi
if want sweep; then group_start
  SW="--epochs 20 --no-final"
  for LR in 0.05 0.1 0.2 0.4 0.8 1.2 1.6 2.4; do job --head affine --lr $LR $SW --out LR_affine_bn_$LR; done
  for LR in 0.05 0.1 0.2 0.4 0.5 0.6 0.7 0.8; do job --head conic --eps 0.5 --lr $LR $SW --out LR_conic_bn_$LR; done
  for LR in 0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64; do
    job --head affine --no-bn --lr $LR $SW             --out LR_affine_nobn_$LR
    job --head conic --eps 0.5 --no-bn --lr $LR $SW    --out LR_conic_nobn_$LR
  done
fi
if want ablation; then group_start
  job --head affine                                    --out BASE_affine
  job --head conic                                     --out cifar10_resnet18_conic_s0
  job --head conic --eps 0.5 --tau-floor 1.0           --out A_floor1_eps05
  job --head conic --eps 0.5 --tau-freeze 5            --out F_taufreeze5
  job --head affine --no-bn --lr 0.01                  --out E_affine_nobn
  job --head conic --eps 0.5 --no-bn --lr 0.01         --out C_nobn_freetau_eps05
  job --dataset cifar100 --head conic --eps 0.5        --out D_cifar100_freetau
fi
if want resnet56; then group_start
  job --backbone resnet56 --head affine                --out R56_skip
  job --backbone resnet56 --head affine --no-skip      --out R56_noskip
fi
if want nwd; then group_start
  for S in 0 1 2 3 4; do
    job --head conic --eps 0.5 --wd-eta 0 --lr 0.1           --seed $S --out NWD_conic_bn_s$S
    job --head conic --eps 0.5 --wd-eta 0 --no-bn --lr 0.01  --seed $S --out NWD_conic_nobn_s$S
    job --head conic --eps 0.5 --wd-eta 0 --lr 0.01          --seed $S --out NWD_conic_bn_lr001_s$S
    job --head conic --eps 0.5 --wd-eta 0 --no-bn --lr 0.08  --seed $S --out NWD_conic_nobn_lr008_s$S
  done
fi
if want proto; then group_start
  for S in 0 1 2 3 4; do
    job --head proto --eps 0.5 --lr 0.1                --seed $S --out PROTO_bn_s$S
    job --head proto --eps 0.5 --no-bn --lr 0.08       --seed $S --out PROTO_nobn_lr008_s$S
  done
fi
if want cosine; then group_start
  for S in 0 1 2 3 4; do
    job --head cosine --lr 0.1                         --seed $S --out COS_bn_s$S
    job --head cosine --no-bn --lr 0.01                --seed $S --out COS_nobn_s$S
    job --head cosine --no-bn --lr 0.08                --seed $S --out COS_nobn_lr008_s$S
  done
fi
if want replicate; then group_start
  for S in 0 1 2 3 4; do
    job --head affine --lr 0.1                         --seed $S --out G_affine_bn_s$S
    job --head conic --eps 0.5 --lr 0.1                --seed $S --out G_conic_bn_s$S
    job --head affine --no-bn --lr 0.01                --seed $S --out G_affine_nobn_s$S
    job --head conic --eps 0.5 --no-bn --lr 0.01       --seed $S --out G_conic_nobn_s$S
    job --head affine --no-bn --lr 0.08                --seed $S --out G_affine_nobn_lr008_s$S
    job --head conic --eps 0.5 --no-bn --lr 0.08       --seed $S --out G_conic_nobn_lr008_s$S
    job --head conic --eps 0.5 --lr 0.01               --seed $S --out G_conic_bn_lr001_s$S
  done
fi
if want c100; then group_start
  for S in 0 1 2 3 4; do
    job --dataset cifar100 --head affine --lr 0.1           --seed $S --out C100_affine_bn_s$S
    job --dataset cifar100 --head conic --eps 0.5 --lr 0.1  --seed $S --out C100_conic_bn_s$S
  done
fi

N=${#JOBS[@]}
if [ "${LIST:-0}" = 1 ]; then          # print the commands and stop
  for j in "${JOBS[@]}"; do echo "$PY src/train.py $BASE $j $EXTRA"; done
  echo "$N runs"; exit 0
fi
echo "$(date '+%F %T')  $N runs, groups:$WANT| $LANES lanes, $WORKERS loader processes per run, $NCPU CPU cores, $PY"

if [ "$QUICK" != 1 ]; then          # fetch the datasets once, before the lanes start
  NEED="CIFAR10"; for j in "${JOBS[@]}"; do [[ "$j" == *cifar100* ]] && NEED="CIFAR10 CIFAR100"; done
  "$PY" -c "
from torchvision import datasets
for name in '$NEED'.split():
    for train in (True, False):
        getattr(datasets, name)('./data', train=train, download=True)
" || { echo "Could not read or download CIFAR into ./data (see README, Data)."; exit 1; }
fi

lane() {
  local L=$1 i name
  for (( i=L; i<N; i+=LANES )); do
    name=${JOBS[$i]##*--out }
    if [ -f "$RUNS/$name/DONE" ]; then echo "$(date '+%F %T')  lane $L  skip   $name (finished)"; continue; fi
    echo "$(date '+%F %T')  lane $L  start  $name"
    "$PY" src/train.py $BASE ${JOBS[$i]} $EXTRA --runs-dir "$RUNS" >> "$LOGS/train_lane$L.log" 2>&1
    echo "$(date '+%F %T')  lane $L  end    $name  ($(cat "$RUNS/$name/DONE" 2>/dev/null || echo "not finished, see $LOGS/train_lane$L.log"))"
  done
}

for (( L=0; L<LANES; L++ )); do
  lane $L &
  [ "$QUICK" = 1 ] || sleep 20      # stagger the starts, so that the first data loading does not collide
done
wait
echo "$(date '+%F %T')  all lanes finished."
for p in S_ CTRL_ LR8_ LR_ BASE_ cifar10_ A_ F_ E_ C_ D_ R56_ NWD_ PROTO_ COS_ G_ C100_; do
  n=$(find "$RUNS" -maxdepth 2 -path "$RUNS/${p}*/DONE" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "  finished ${p}* runs: $n"
done
exit 0
