#!/bin/bash

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
export OMP_NUM_THREADS=1

# MIOpen tuning cache - runs once, reused forever after
export MIOPEN_USER_DB_PATH=/users/arivajoo/.miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=/users/arivajoo/.miopen_cache
mkdir -p /users/arivajoo/.miopen_cache

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export MIOPEN_ENABLE_LOGGING=0

singularity run $SIF \
  /users/arivajoo/joonas-env/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=1 \
  src/data/make_val_split.py
