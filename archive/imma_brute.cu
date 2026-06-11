// imma_brute.cu — Brute-force find IMMA fragment layout.
// Strategy: dump raw IMMA output (128 s32 values) and compare against
// CPU reference to find the mapping.

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

// Dump raw output: each thread writes 4 C values to C_out[thread*4+0..3]
// AND dump A frag (4 regs) and B frag (2 regs) for verification
__global__ void brute_test(int32_t* C_raw, uint32_t* A_raw, uint32_t* B_raw,
    const int8_t* A_in, const int8_t* Bt_in, int layout)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];

    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 512; i += 32) ((int8_t*)a_sm)[i] = A_in[i];
    for (int i = threadIdx.x; i < 256; i += 32) ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    uint32_t a0,a1,a2,a3, b0,b1;

    if (layout == 3) {
        // Layout 3: A row=n%16, col=(n/16)*16+reg*4
        int ar = lane % 16;
        int acb = (lane / 16) * 16;
        a0=pack4s8(&a_sm[ar][acb+0]); a1=pack4s8(&a_sm[ar][acb+4]);
        a2=pack4s8(&a_sm[ar][acb+8]); a3=pack4s8(&a_sm[ar][acb+12]);
        // B: col=n%8, k=(n/8)*8+reg*4
        int bc=lane%8, bkb=(lane/8)*8;
        b0=pack4s8(&b_sm[bc][bkb+0]); b1=pack4s8(&b_sm[bc][bkb+4]);
    } else {
        // Layout 5: A row=n/4, col=(n%4)*8+reg*4; +8 rows for reg 2,3
        int ar1=lane/4, ar2=ar1+8, acb=(lane%4)*8;
        a0=pack4s8(&a_sm[ar1][acb+0]); a1=pack4s8(&a_sm[ar1][acb+4]);
        a2=pack4s8(&a_sm[ar2][acb+0]); a3=pack4s8(&a_sm[ar2][acb+4]);
        int bc=lane/4, bkb=(lane%4)*8;
        b0=pack4s8(&b_sm[bc][bkb+0]); b1=pack4s8(&b_sm[bc][bkb+4]);
    }

    // Dump fragments
    A_raw[lane*4+0]=a0; A_raw[lane*4+1]=a1; A_raw[lane*4+2]=a2; A_raw[lane*4+3]=a3;
    B_raw[lane*2+0]=b0; B_raw[lane*2+1]=b1;

    // IMMA
    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3, a0,a1,a2,a3, b0,b1, 0,0,0,0);

    // Raw dump: thread n -> C_raw[n*4+0..3]
    C_raw[lane*4+0]=c0; C_raw[lane*4+1]=c1;
    C_raw[lane*4+2]=c2; C_raw[lane*4+3]=c3;
}

int main() {
    const int M=16, N=8, K=32;

    // Use UNIQUE values: A[i][k] = i*32+k+1 (1..512), B[k][j] = k*8+j+1 (1..256)
    // Keep them small enough for int8: use values 1..127
    int8_t h_A[512], h_Bt[256];
    for (int i=0;i<M;i++)
        for (int k=0;k<K;k++)
            h_A[i*K+k] = (int8_t)((i*32+k) % 127 + 1);  // 1..127, unique per (i,k)
    for (int j=0;j<N;j++)
        for (int k=0;k<K;k++)
            h_Bt[j*K+k] = (int8_t)((j*32+k) % 127 + 1);  // 1..127, unique per (j,k)

    // CPU reference: C[i][j] = sum_k A[i][k] * B[k][j]
    // B[k][j] = h_Bt[j][k] = h_Bt[j*K+k]
    int32_t h_Cref[128]; memset(h_Cref,0,sizeof(h_Cref));
    for (int i=0;i<M;i++)
        for (int j=0;j<N;j++)
            for (int k=0;k<K;k++)
                h_Cref[i*N+j] += (int32_t)h_A[i*K+k] * (int32_t)h_Bt[j*K+k];

    int8_t *d_A,*d_Bt; int32_t *d_C; uint32_t *d_Af,*d_Bf;
    cudaMalloc(&d_A,512); cudaMalloc(&d_Bt,256);
    cudaMalloc(&d_C,128*4); cudaMalloc(&d_Af,128*4); cudaMalloc(&d_Bf,64*4);
    cudaMemcpy(d_A,h_A,512,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bt,h_Bt,256,cudaMemcpyHostToDevice);

    printf("IMMA brute-force fragment layout finder\n");
    printf("=======================================\n\n");

    for (int layout = 3; layout <= 5; layout += 2) {
        printf("--- Layout %d ---\n", layout);
        brute_test<<<1,32>>>(d_C, d_Af, d_Bf, d_A, d_Bt, layout);
        cudaDeviceSynchronize();

        int32_t h_C[128]; uint32_t h_Af[128], h_Bf[64];
        cudaMemcpy(h_C, d_C, 128*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Af, d_Af, 128*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Bf, d_Bf, 64*4, cudaMemcpyDeviceToHost);

        // Count how many GPU values appear in CPU reference
        int found = 0;
        for (int t = 0; t < 128; t++) {
            for (int r = 0; r < 128; r++) {
                if (h_C[t] == h_Cref[r]) { found++; break; }
            }
        }
        printf("  GPU values found in CPU ref: %d/128\n", found);

        if (found > 100) {
            // Most values match -> layout is correct, just need C mapping
            printf("  C mapping (thread -> (row,col)):\n");
            for (int t = 0; t < 32; t++) {
                for (int r = 0; r < 4; r++) {
                    int gv = h_C[t*4+r];
                    for (int i = 0; i < M; i++)
                        for (int j = 0; j < N; j++)
                            if (gv == h_Cref[i*N+j])
                                printf("    T%d R%d -> C[%d][%d]=%d\n", t, r, i, j, gv);
                }
            }
        }

        // Print first 4 threads' C values for inspection
        printf("  First 4 threads C:\n");
        for (int t = 0; t < 4; t++)
            printf("    T%2d: %8d %8d %8d %8d\n", t, h_C[t*4], h_C[t*4+1], h_C[t*4+2], h_C[t*4+3]);
    }

    // Print CPU ref for comparison
    printf("\nCPU ref C[16][8]:\n");
    for (int i=0;i<M;i++){
        printf("  [%2d]", i);
        for(int j=0;j<N;j++) printf(" %8d", h_Cref[i*N+j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_Bt); cudaFree(d_C);
    cudaFree(d_Af); cudaFree(d_Bf);
    return 0;
}
