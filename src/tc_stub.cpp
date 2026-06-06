// tc_stub.cpp — fallback for tc_jackpot_search when the CUTLASS tensor-core path
// is not built (GPU < sm_80 or CUTLASS headers absent). Lets plainproof_gen link
// and run its dp4a (CUDA-core) path; the `--tc` option then reports "no win".
#include <cstdio>

extern "C" int tc_jackpot_search(
    const signed char* /*a_noised*/, const signed char* /*b_noised_t*/,
    int /*m*/, int /*n*/, int /*k*/, int /*rank*/,
    const int* /*pat*/, int /*h*/, int /*w*/,
    const int* /*row_off*/, int /*nrow_off*/, const int* /*col_off*/, int /*ncol_off*/,
    const unsigned char* /*a_noise_seed32*/, const unsigned char* /*bound_le32*/,
    unsigned int* /*out_hashes_host*/, unsigned int* /*dbg_j0_host*/,
    int* out_rt, int* out_ct)
{
    if (out_rt) *out_rt = -1;
    if (out_ct) *out_ct = -1;
    fprintf(stderr, "tc_jackpot_search: STUB (tensor-core path not built; use dp4a --mine/--gpu)\n");
    return -1;
}
