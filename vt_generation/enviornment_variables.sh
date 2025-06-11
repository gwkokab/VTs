#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export NPROC=16
#export OPENBLAS_NUM_THREADS=1
#export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

#export intra_op_parallelism_threads=1
#export TF_CPP_MIN_LOG_LEVEL=0
#export XLA_PYTHON_CLIENT_ALLOCATOR=platform
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export TF_FORCE_GPU_ALLOW_GROWTH=false
#export MKL_NUM_THREADS=1
#export OMP_NUM_THREADS=1

#Comment to run in gpu
#export JAX_PLATFORMS=cpu

# JAX Configuration
export JAX_COMPILATION_CACHE_DIR="$HOME/jax_cache"

#for higher precision uncomment it.
export JAX_ENABLE_X64=1

# Use JAX backend for wcosmo
export WCOSMO_ARRAY_API=jax
