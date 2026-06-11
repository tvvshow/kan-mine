// imma_verify.cu — Micro-test for mma.sync.m16n8k32 fragment layout.
// Single warp does 16×32 × 32×8 = 16×8 GEMM, compare GPU vs CPU.
// Upload to box, compile: nvcc -O3 -arch=sm_86 -o imma_verify imma_verify.cu
// Run: ./imma_verify

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

static __device__ __forceinline__ uint32_t pack4s8(const int8_t* p) {
    return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1]<<8)
         | ((uint32_t)(uint8_t)p[2]<<16) | ((uint32_t)(uint8_t)p[3]<<24);
}

static __device__ void imma16832(
    int &d0, int &d1, int &d2, int &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int c0, int c1, int c2, int c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n\t"
        "  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}

// Test: 1 warp, 16×32 × 32×8 GEMM
// A[16][32] row-major, B[32][8] col-major stored as Bt[8][32] row-major
// C[16][8] int32 output
__global__ void imma_test_kernel(int32_t* C_out,
                                  const int8_t* A_in, const int8_t* Bt_in) {
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];

    int lane = threadIdx.x & 31;

    // Load A into smem (512 bytes)
    for (int i = threadIdx.x; i < 512; i += 32)
        ((int8_t*)a_sm)[i] = A_in[i];
    // Load Bt into smem (256 bytes)
    for (int i = threadIdx.x; i < 256; i += 32)
        ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    // --- A fragment: thread n holds ---
    //   a[0]: row (n/4),   cols (n%4)*8 + 0..3
    //   a[1]: row (n/4),   cols (n%4)*8 + 4..7
    //   a[2]: row (n/4)+8, cols (n%4)*8 + 0..3
    //   a[3]: row (n/4)+8, cols (n%4)*8 + 4..7
    int ar1 = lane / 4;
    int ar2 = lane / 4 + 8;
    int acb = (lane % 4) * 8;
    uint32_t a0 = pack4s8(&a_sm[ar1][acb]);
    uint32_t a1 = pack4s8(&a_sm[ar1][acb + 4]);
    uint32_t a2 = pack4s8(&a_sm[ar2][acb]);
    uint32_t a3 = pack4s8(&a_sm[ar2][acb + 4]);

    // --- B fragment (col-major B, stored as Bt row-major in b_sm) ---
    //   b[0]: B[(n%4)*8+0..3, n/4] = b_sm[n/4][(n%4)*8+0..3]
    //   b[1]: B[(n%4)*8+4..7, n/4] = b_sm[n/4][(n%4)*8+4..7]
    int bc = lane / 4;       // 0..7 → column of B = row of b_sm
    int bkb = (lane % 4) * 8;
    uint32_t b0 = pack4s8(&b_sm[bc][bkb]);
    uint32_t b1 = pack4s8(&b_sm[bc][bkb + 4]);

    // Run IMMA
    int c0=0, c1=0, c2=0, c3=0;
    imma16832(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    // --- Store C fragment ---
    //   c[0]: C[n/4,   (n%4)*2]
    //   c[1]: C[n/4,   (n%4)*2+1]
    //   c[2]: C[n/4+8, (n%4)*2]
    //   c[3]: C[n/4+8, (n%4)*2+1]
    int cr1 = lane / 4;
    int cr2 = lane / 4 + 8;
    int cc1 = (lane % 4) * 2;
    int cc2 = (lane % 4) * 2 + 1;

    C_out[cr1 * 8 + cc1] = c0;
    C_out[cr1 * 8 + cc2] = c1;
    C_out[cr2 * 8 + cc1] = c2;
    C_out[cr2 * 8 + cc2] = c3;
}

int main() {
    const int M=16, N=8, K=32;

    // Fill A with known values (avoid overflow: keep values small)
    int8_t h_A[512];
    for (int i = 0; i < M*K; i++) h_A[i] = (int8_t)((i * 3 + 1) % 11 - 5);

    // Fill Bt (B transposed, 8×32 row-major)
    // B[k][n] = Bt[n][k]
    int8_t h_Bt[256];
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            h_Bt[n*K+k] = (int8_t)((n*K+k * 7 + 3) % 9 - 4);

    // CPU reference: C[i][j] = sum_k A[i][k] * B[k][j]
    // B[k][j] = h_Bt[j*K+k]
    int32_t h_Cref[128];
    memset(h_Cref, 0, sizeof(h_Cref));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int kk = 0; kk < K; kk++)
                h_Cref[i*N+j] += (int32_t)h_A[i*K+kk] * (int32_t)h_Bt[j*K+kk];

    // GPU
    int8_t *d_A, *d_Bt;
    int32_t *d_C;
    cudaMalloc(&d_A, 512);
    cudaMalloc(&d_Bt, 256);
    cudaMalloc(&d_C, 128*4);
    cudaMemcpy(d_A, h_A, 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bt, h_Bt, 256, cudaMemcpyHostToDevice);

    imma_test_kernel<<<1, 32>>>(d_C, d_A, d_Bt);
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int32_t h_Cgpu[128];
    cudaMemcpy(h_Cgpu, d_C, 128*4, cudaMemcpyDeviceToHost);

    // Compare
    int errs = 0;
    printf("IMMA m16n8k32 fragment layout test\n");
    printf("==================================\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N+j;
            if (h_Cref[idx] != h_Cgpu[idx]) {
                if (errs < 20)
                    printf("  MISS [%d][%d]: CPU=%d GPU=%d\n", i, j, h_Cref[idx], h_Cgpu[idx]);
                errs++;
            }
        }
    }
    if (errs == 0) printf("ALL 128 VALUES MATCH! Fragment layout is CORRECT.\n");
    else printf("%d / 128 mismatches. Layout WRONG.\n", errs);

    // Print both for visual comparison
    printf("\nCPU reference (first 8 rows):\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < N; j++) printf("%6d", h_Cref[i*N+j]);
        printf("\n");
    }
    printf("\nGPU result (first 8 rows):\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < N; j++) printf("%6d", h_Cgpu[i*N+j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_Bt); cudaFree(d_C);
    return errs ? 1 : 0;
}
