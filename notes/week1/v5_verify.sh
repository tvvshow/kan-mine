#!/usr/bin/env bash
# v5 (async search/prep overlap) build + verify + bench, one shot.
# Run from ~/peral/build. Assumes GPU is FREE (check nvidia-smi first!).
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd ~/peral/build

echo "=== [1/5] compile (async overlap sources) ==="
nvcc -O3 -arch=sm_86 -std=c++17 -I"$HOME/cutlass/include" -c ../src/tc_cutlass_v2.cu -o tc_cutlass2_async.o
nvcc -O3 -arch=sm_86 -std=c++17 -c ../src/gpu_prep.cu -o gpu_prep_async.o
g++ -O3 -std=c++17 -fopenmp -I../blake3 -I/usr/local/cuda/include -c ../src/plainproof_gen.cpp -o plainproof_gen_async.o
g++ -O3 -std=c++17 -fopenmp -I../blake3 -I/usr/local/cuda/include -DPROVER_LIB -c ../src/plainproof_gen.cpp -o prover_lib_async.o

echo "=== [2/5] link: CLI (POSTCHECK harness) + pearl-miner-v5 ==="
BL_OBJ="blake3.o blake3_dispatch.o blake3_portable.o blake3_sse2_x86-64_unix.o blake3_sse41_x86-64_unix.o blake3_avx2_x86-64_unix.o blake3_avx512_x86-64_unix.o"
g++ -O3 -fopenmp plainproof_gen_async.o $BL_OBJ tc_cutlass2_async.o gpu_prep_async.o \
  -L/usr/local/cuda/lib64 -lcudart -o plainproof_gen_v5
g++ -O3 -fopenmp miner_main.o prover_lib_async.o $BL_OBJ tc_cutlass2_async.o gpu_prep_async.o \
  -L/usr/local/cuda/lib64 -lcudart -lssl -lcrypto -lpthread -o pearl-miner-v5
echo BUILD_OK

echo "=== [3/5] CORRECTNESS: loose target 2^216 -> must WIN draw 1-ish, POSTCHECK ok=1 ==="
./plainproof_gen_v5 --cfg real --mine 4 --tc \
  --target 0000000001000000000000000000000000000000000000000000000000000000 \
  2>&1 | grep -E "MINE WIN|POSTCHECK|async|GPUPREP|tc\(cutlass2\)" || true
echo "---- gate: expect 'POSTCHECK ... ok=1' above ----"

echo "=== [4/5] THROUGHPUT: impossible target, 30 draws, wall-clock draws/sec ==="
./plainproof_gen_v5 --cfg real --mine 30 --tc \
  --target 0000000000000000000000000000000000000000000000000000000000000001 \
  2>&1 | grep -E "draws/sec|avg|async|tc\(cutlass2\)" | tail -8

echo "=== [5/5] summary ==="
echo "v4 reference: kernel 650ms/108 TH/s, wall 102.5 TH/s."
echo "v5 target   : wall ~= kernel (107-108 TH/s). If POSTCHECK ok=1 AND faster -> go live."
