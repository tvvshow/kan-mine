// tc_cutlass_small.cu — BM=64 BN=128 variant for 2 TB/SM occupancy test
// Expected: 2× occupancy → 30-50% speedup (260 → 350+ TH/s kernel)
//
// Recipe change from tc_cutlass_v2.cu:
//   TBShape 128×256×64 → 64×128×64
//   WarpShape 64×64×64 → 32×64×64  (must evenly divide TB)
//
// Everything else identical: GROUPM, fold callback, lane-distributed redux.
// Build: nvcc -O3 -arch=sm_89 -std=c++17 -DSMALL_TILE -I$CUTLASS_HOME/include \
//        -c src/tc_cutlass_small.cu -o build/tc_kernel.o

#define SMALL_TILE
#include "tc_cutlass_v2.cu"
