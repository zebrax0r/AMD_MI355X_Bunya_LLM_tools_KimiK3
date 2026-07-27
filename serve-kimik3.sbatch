#!/bin/bash --login
#SBATCH --job-name=kimik3-mi355x
#SBATCH --partition=admin_test
#SBATCH --account=a_rcc
#SBATCH --qos=sdf
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1800G
#SBATCH --time=48:00:00
#SBATCH --output=logs/kimik3-%j.out
#SBATCH --error=logs/kimik3-%j.out
#
# Batch wrapper around serve-kimik3.sh for Bunya's MI355X nodes (bun159/160/161).
# Submit from /scratch (not /home, not RDM) with:
#
#   mkdir -p logs && sbatch serve-kimik3.sbatch
#
# The partition/account/qos/gres above are the discovered MI355X recipe:
#   --partition=admin_test  (the MI355X nodes live here, not gpu_rocm)
#   --account=a_rcc  --qos=sdf  --gres=gpu:mi355x:8
# The MI355X nodes are bun159/bun160/bun161; the gres lands you on a free one.
# Add e.g. '#SBATCH --nodelist=bun161' if you want to pin a specific node.
# Confirm the current recipe with:  sinfo -o "%P %.10l %G %f" | grep mi355x
#
# Walltime note: a cold Kimi K3 start reads ~1.5 TB of weights off scratch and
# JIT-compiles FP4 kernels on first run, so budget a substantial slice of the
# allocation for startup alone. READY_TIMEOUT in kimik3.env (default 14400s)
# must stay below this walltime.
#
# Memory: --mem=1800G is host RAM for staging the weight load, not HBM.
#
# The connection banner (node, endpoint URL, API key location, tunnel command)
# is printed into the job log. Once the job is RUNNING:
#
#   grep -A 24 'is up and serving' logs/kimik3-<jobid>.out
#
# The server runs until walltime or 'scancel <jobid>' (the script traps SIGTERM
# and stops the container cleanly).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[kimik3-sbatch] Job $SLURM_JOB_ID on $(hostname) at $(date)"
exec ./serve-kimik3.sh serve
