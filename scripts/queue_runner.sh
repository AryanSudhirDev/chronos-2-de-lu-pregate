#!/bin/zsh
# Runs commands from runs/queue_<name>.txt one at a time, first line first, until runs/STOP exists.
# Lines can be appended or edited while it runs. Log: runs/queue_<name>.log
# Start detached:  screen -dmS q_<name> scripts/queue_runner.sh <name>
cd "$(dirname "$0")/.."
name=${1:-gpu}
q=runs/queue_$name.txt
log=runs/queue_$name.log
touch $q
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
while [[ ! -e runs/STOP ]]; do
  cmd=$(head -n 1 $q)
  if [[ -z "$cmd" ]]; then
    sed -i '' '1d' $q 2>/dev/null
    sleep 30
    continue
  fi
  sed -i '' '1d' $q
  echo "=== $(date '+%F %T') START $cmd" >> $log
  zsh -c "$cmd" >> $log 2>&1
  rc=$?
  echo "=== $(date '+%F %T') END rc=$rc $cmd" >> $log
  echo "$(date '+%F %T') [$name] rc=$rc $cmd" >> runs/queue_done.txt
done
