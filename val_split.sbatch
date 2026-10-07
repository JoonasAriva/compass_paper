#!/bin/bash
#SBATCH --job-name=val_split
#SBATCH --account=project_465002884
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=val_splitting_%j.log

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

chmod +x ./run_val_split.sh
srun ./run_val_split.sh