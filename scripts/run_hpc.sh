#!/bin/bash
# =============================================================================
# run_hpc.sh — SGE job submission script for BU Shared Computing Cluster (SCC)
#
# NOTE: This script is specific to Boston University's SCC (scc.bu.edu).
#       If you are running on a different cluster, adapt the conda activation
#       path, the project directory, and the SGE directives accordingly.
#
# Submit with:
#   qsub scripts/run_hpc.sh
# =============================================================================

set -euo pipefail

# Job name shown in the queue
#$ -N mice_lfp_decoder

# Maximum wall-clock time (hh:mm:ss)
#$ -l h_rt=24:00:00

# RAM per core (96 cores × 8 GB = up to 768 GB total addressable)
#$ -l mem_per_core=8G

# Number of OpenMP threads / CPU cores
#$ -pe omp 26

# Redirect stdout and stderr
#$ -o gru_transformer_out.log
#$ -e gru_transformer_err.log

# Activate the shared PyTorch conda environment on BU SCC
source /share/pkg.8/miniconda/25.3.1/install/etc/profile.d/conda.sh
conda activate /share/pkg.8/academic-ml/fall-2025/install/fall-2025-pyt

# Move to the project root (where config.yaml lives)
cd /projectnb/cs523aw/students/pangelos/

python3 scripts/train_gru_transformer.py --config config.yaml
