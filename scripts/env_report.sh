#!/usr/bin/env bash
# scripts/env_report.sh -- record the software and hardware of this machine.
#
#   bash scripts/env_report.sh            prints the report and writes ENVIRONMENT.txt
#
# Run it once on the machine that produced the results and keep ENVIRONMENT.txt in
# the repository, so that readers can see the exact versions behind the numbers.

set -u
cd "$(dirname "$0")/.."
PY=${PY:-}
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
{
  echo "date: $(date '+%F %T %Z')"
  echo "os: $(uname -srm)"
  [ -f /etc/os-release ] && echo "distribution: $(. /etc/os-release; echo "$PRETTY_NAME")"
  echo "cpu cores: $(getconf _NPROCESSORS_ONLN 2>/dev/null)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  fi
  "$PY" - <<'EOF'
import sys, platform
print('python:', sys.version.split()[0])
for m in ('torch', 'torchvision', 'numpy', 'scipy', 'matplotlib'):
    try:
        mod = __import__(m); print('%s: %s' % (m, mod.__version__))
    except Exception:
        print('%s: not installed' % m)
import torch
print('cuda (torch build):', torch.version.cuda)
print('cudnn:', torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'none')
print('cuda available:', torch.cuda.is_available())
EOF
  command -v octave >/dev/null 2>&1 && echo "octave: $(octave --version 2>/dev/null | head -1)"
} | tee ENVIRONMENT.txt
echo "(written to ENVIRONMENT.txt)"
