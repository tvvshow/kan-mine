// imma_verify2.cu — Verify IMMA fragment layout using ldmatrix for loading.
// Tests multiple ldmatrix configurations to find the correct one.
//
// Key insight: ldmatrix.x2.b16 with stride=32 bytes matches our a_sh[BM][32] layout.
// For 16-row A: 2 calls of ldmatrix.x2 (rows 0-7, rows 8-15) → 4 regs.
// For 8-row B^T: 1 call of ldmatrix.x2 → 2 regs.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

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

// Variant A: ldmatrix.x2.b16 for A (2 calls), ldmatrix.x2.b16 for B (1 call, NO trans)
__global__ void testA_ldmatrix_notrans(int32_t* C_out,
    const int8_t* A_in, const int8_t* Bt_in)
{
    __shared__ int8_t a_sm[16][32];  // A row-major
    __shared__ int8_t b_sm[8][32];   // B^T row-major

    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 512; i += 32) ((int8_t*)a_sm)[i] = A_in[i];
    for (int i = threadIdx.x; i < 256; i += 32) ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    // Load A: ldmatrix.x2.b16, call 1 (rows 0-7), call 2 (rows 8-15)
    uint32_t a0, a1, a2, a3;
    uint32_t aptr1 = (uint32_t)__cvta_generic_to_shared(&a_sm[0][0]);
    uint32_t aptr2 = (uint32_t)__cvta_generic_to_shared(&a_sm[8][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(a0), "=r"(a1) : "r"(aptr1));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(a2), "=r"(a3) : "r"(aptr2));

    // Load B: ldmatrix.x2.b16 NO trans
    uint32_t b0, b1;
    uint32_t bptr = (uint32_t)__cvta_generic_to_shared(&b_sm[0][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(b0), "=r"(b1) : "r"(bptr));

    // IMMA
    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3, a0,a1,a2,a3, b0,b1, 0,0,0,0);

    // Store C: try layout row=(n/4), col=(n%4)*2 + {0,1}, row+8
    int cr1=lane/4, cr2=lane/4+8, cc1=(lane%4)*2, cc2=(lane%4)*2+1;
    C_out[cr1*8+cc1]=c0; C_out[cr1*8+cc2]=c1;
    C_out[cr2*8+cc1]=c2; C_out[cr2*8+cc2]=c3;
}

// Variant B: same A loading, B with .trans
__global__ void testB_ldmatrix_trans(int32_t* C_out,
    const int8_t* A_in, const int8_t* Bt_in)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];

    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 512; i += 32) ((int8_t*)a_sm)[i] = A_in[i];
    for (int i = threadIdx.x; i < 256; i += 32) ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    uint32_t aptr1 = (uint32_t)__cvta_generic_to_shared(&a_sm[0][0]);
    uint32_t aptr2 = (uint32_t)__cvta_generic_to_shared(&a_sm[8][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(a0), "=r"(a1) : "r"(aptr1));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(a2), "=r"(a3) : "r"(aptr2));

    uint32_t b0, b1;
    uint32_t bptr = (uint32_t)__cvta_generic_to_shared(&b_sm[0][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                 : "=r"(b0), "=r"(b1) : "r"(bptr));

    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3, a0,a1,a2,a3, b0,b1, 0,0,0,0);

    int cr1=lane/4, cr2=lane/4+8, cc1=(lane%4)*2, cc2=(lane%4)*2+1;
    C_out[cr1*8+cc1]=c0; C_out[cr1*8+cc2]=c1;
    C_out[cr2*8+cc1]=c2; C_out[cr2*8+cc2]=c3;
}

// Variant C: A with .trans, B without .trans
__global__ void testC_Atrans_Bnotrans(int32_t* C_out,
    const int8_t* A_in, const int8_t* Bt_in)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];

    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 512; i += 32) ((int8_t*)a_sm)[i] = A_in[i];
    for (int i = threadIdx.x; i < 256; i += 32) ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    uint32_t aptr1 = (uint32_t)__cvta_generic_to_shared(&a_sm[0][0]);
    uint32_t aptr2 = (uint32_t)__cvta_generic_to_shared(&a_sm[8][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                 : "=r"(a0), "=r"(a1) : "r"(aptr1));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                 : "=r"(a2), "=r"(a3) : "r"(aptr2));

    uint32_t b0, b1;
    uint32_t bptr = (uint32_t)__cvta_generic_to_shared(&b_sm[0][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(b0), "=r"(b1) : "r"(bptr));

    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3, a0,a1,a2,a3, b0,b1, 0,0,0,0);

    int cr1=lane/4, cr2=lane/4+8, cc1=(lane%4)*2, cc2=(lane%4)*2+1;
    C_out[cr1*8+cc1]=c0; C_out[cr1*8+cc2]=c1;
    C_out[cr2*8+cc1]=c2; C_out[cr2*8+cc2]=c3;
}

// Variant D: dump raw ldmatrix fragment to see what each thread gets
// Each thread writes its A fragment (4 regs) and B fragment (2 regs) directly
__global__ void testD_dump_fragments(uint32_t* A_frag_out, uint32_t* B_frag_out,
    const int8_t* A_in, const int8_t* Bt_in, int variant)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];

    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 512; i += 32) ((int8_t*)a_sm)[i] = A_in[i];
    for (int i = threadIdx.x; i < 256; i += 32) ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    uint32_t a0, a1, a2, a3, b0, b1;

    if (variant == 0) {
        // ldmatrix.x2 no trans
        uint32_t aptr1 = (uint32_t)__cvta_generic_to_shared(&a_sm[0][0]);
        uint32_t aptr2 = (uint32_t)__cvta_generic_to_shared(&a_sm[8][0]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                     : "=r"(a0), "=r"(a1) : "r"(aptr1));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                     : "=r"(a2), "=r"(a3) : "r"(aptr2));
        uint32_t bptr = (uint32_t)__cvta_generic_to_shared(&b_sm[0][0]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                     : "=r"(b0), "=r"(b1) : "r"(bptr));
    }

    A_frag_out[lane*4+0]=a0; A_frag_out[lane*4+1]=a1;
    A_frag_out[lane*4+2]=a2; A_frag_out[lane*4+3]=a3;
    B_frag_out[lane*2+0]=b0; B_frag_out[lane*2+1]=b1;
}

static int count_mismatches(const int32_t* ref, const int32_t* gpu, int n) {
    int errs = 0;
    for (int i = 0; i < n; i++) if (ref[i] != gpu[i]) errs++;
    return errs;
}

int main() {
    const int M=16, N=8, K=32;

    // Use simple distinct values
    int8_t h_A[512], h_Bt[256];
    for (int i = 0; i < M*K; i++) h_A[i] = (int8_t)((i*3+1) % 11 - 5);
    for (int i = 0; i < N*K; i++) h_Bt[i] = (int8_t)((i*7+3) % 9 - 4);

    // CPU reference
    int32_t h_Cref[128];
    memset(h_Cref, 0, sizeof(h_Cref));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int kk = 0; kk < K; kk++)
                h_Cref[i*N+j] += (int32_t)h_A[i*K+kk] * (int32_t)h_Bt[j*K+kk];

    int8_t *d_A, *d_Bt;
    int32_t *d_C;
    uint32_t *d_Af, *d_Bf;
    cudaMalloc(&d_A, 512); cudaMalloc(&d_Bt, 256);
    cudaMalloc(&d_C, 128*4); cudaMalloc(&d_Af, 128*4); cudaMalloc(&d_Bf, 64*4);
    cudaMemcpy(d_A, h_A, 512, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bt, h_Bt, 256, cudaMemcpyHostToDevice);

    printf("IMMA m16n8k32 fragment layout test (ldmatrix variants)\n");
    printf("======================================================\n\n");

    // --- Test A: ldmatrix no trans ---
    testA_ldmatrix_notrans<<<1,32>>>(d_C, d_A, d_Bt);
    cudaDeviceSynchronize();
    int32_t h_Cgpu[128];
    cudaMemcpy(h_Cgpu, d_C, 128*4, cudaMemcpyDeviceToHost);
    int errsA = count_mismatches(h_Cref, h_Cgpu, 128);
    printf("Test A (ldmatrix.x2 no trans for both A,B): %d/128 mismatches %s\n",
           errsA, errsA==0 ? "✓ CORRECT!" : "✗ WRONG");

    // --- Test B: B with .trans ---
    testB_ldmatrix_trans<<<1,32>>>(d_C, d_A, d_Bt);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Cgpu, d_C, 128*4, cudaMemcpyDeviceToHost);
    int errsB = count_mismatches(h_Cref, h_Cgpu, 128);
    printf("Test B (ldmatrix.x2 A no-trans, B .trans): %d/128 mismatches %s\n",
           errsB, errsB==0 ? "✓ CORRECT!" : "✗ WRONG");

    // --- Test C: A with .trans ---
    testC_Atrans_Bnotrans<<<1,32>>>(d_C, d_A, d_Bt);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Cgpu, d_C, 128*4, cudaMemcpyDeviceToHost);
    int errsC = count_mismatches(h_Cref, h_Cgpu, 128);
    printf("Test C (ldmatrix.x2 A .trans, B no-trans): %d/128 mismatches %s\n",
           errsC, errsC==0 ? "✓ CORRECT!" : "✗ WRONG");

    // --- Dump fragments ---
    testD_dump_fragments<<<1,32>>>(d_Af, d_Bf, d_A, d_Bt, 0);
    cudaDeviceSynchronize();
    uint32_t h_Af[128], h_Bf[64];
    cudaMemcpy(h_Af, d_Af, 128*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Bf, d_Bf, 64*4, cudaMemcpyDeviceToHost);

    printf("\n--- Fragment dump (ldmatrix.x2 no trans, first 8 threads) ---\n");
    printf("Thread | A_frag[0..3] (as 4 s8 each) | B_frag[0..1]\n");
    for (int t = 0; t < 8; t++) {
        printf("  %2d   | ", t);
        for (int r = 0; r < 4; r++) {
            uint32_t v = h_Af[t*4+r];
            printf("[%d,%d,%d,%d] ", (int8_t)(v&0xFF), (int8_t)((v>>8)&0xFF),
                   (int8_t)((v>>16)&0xFF), (int8_t)((v>>24)&0xFF));
        }
        printf("| ");
        for (int r = 0; r < 2; r++) {
            uint32_t v = h_Bf[t*2+r];
            printf("[%d,%d,%d,%d] ", (int8_t)(v&0xFF), (int8_t)((v>>8)&0xFF),
                   (int8_t)((v>>16)&0xFF), (int8_t)((v>>24)&0xFF));
        }
        printf("\n");
    }

    // Also show expected A for first few threads
    printf("\nExpected A row-major (first 4 rows, all 32 cols):\n");
    for (int i = 0; i < 4; i++) {
        printf("  row %2d: ", i);
        for (int k = 0; k < 32; k++) printf("%3d ", (int)h_A[i*32+k]);
        printf("\n");
    }

    // Show B^T (first 4 rows)
    printf("\nB^T row-major (first 4 rows, all 32 cols):\n");
    for (int j = 0; j < 4; j++) {
        printf("  row %2d: ", j);
        for (int k = 0; k < 32; k++) printf("%3d ", (int)h_Bt[j*32+k]);
        printf("\n");
    }

    printf("\nCPU ref C:\n");
    for (int i = 0; i < M; i++) {
        printf("  ");
        for (int j = 0; j < N; j++) printf("%6d", h_Cref[i*N+j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_Bt); cudaFree(d_C);
    cudaFree(d_Af); cudaFree(d_Bf);

    return (errsA==0 || errsB==0 || errsC==0) ? 0 : 1;
}
