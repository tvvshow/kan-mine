// imma_verify3.cu — IMMA test with Layout 3 (row=n%16, col=(n/16)*X)
// A: row=n%16, col=(n/16)*16+0..15 (4 regs of 4 s8)
// B: col=n%8, k=(n/8)*8+0..7 (2 regs of 4 s8)
// C: row=n%16, col=(n/16)*4+0..3 (4 regs of 1 s32)
//
// Also tests Layout 4 as fallback:
// A: row=n/2, col=(n%2)*16+reg*4 (one row per 2 threads, 16 cols each)

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

// Layout 3: A row=n%16, col=(n/16)*16; B col=n%8, k=(n/8)*8; C row=n%16, col=(n/16)*4
__global__ void test_layout3(int32_t* C_out,
    const int8_t* A_in, const int8_t* Bt_in)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];

    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 512; i += 32) ((int8_t*)a_sm)[i] = A_in[i];
    for (int i = threadIdx.x; i < 256; i += 32) ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    // A: row=n%16, col_base=(n/16)*16
    int ar = lane % 16;
    int acb = (lane / 16) * 16;
    uint32_t a0 = pack4s8(&a_sm[ar][acb + 0]);
    uint32_t a1 = pack4s8(&a_sm[ar][acb + 4]);
    uint32_t a2 = pack4s8(&a_sm[ar][acb + 8]);
    uint32_t a3 = pack4s8(&a_sm[ar][acb + 12]);

    // B: col=n%8 (row of b_sm), k_base=(n/8)*8
    int bc = lane % 8;
    int bkb = (lane / 8) * 8;
    uint32_t b0 = pack4s8(&b_sm[bc][bkb + 0]);
    uint32_t b1 = pack4s8(&b_sm[bc][bkb + 4]);

    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3, a0,a1,a2,a3, b0,b1, 0,0,0,0);

    // C: row=n%16, col_base=(n/16)*4
    int cr = lane % 16;
    int ccb = (lane / 16) * 4;
    C_out[cr*8+ccb+0]=c0; C_out[cr*8+ccb+1]=c1;
    C_out[cr*8+ccb+2]=c2; C_out[cr*8+ccb+3]=c3;
}

// Layout 4: A row=n/2, col=(n%2)*16; B col=n%4, k=(n/4)*8; C row=n/2, col=(n%2)*4
__global__ void test_layout4(int32_t* C_out,
    const int8_t* A_in, const int8_t* Bt_in)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];

    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 512; i += 32) ((int8_t*)a_sm)[i] = A_in[i];
    for (int i = threadIdx.x; i < 256; i += 32) ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    // A: row=n/2, col_base=(n%2)*16
    int ar = lane / 2;
    int acb = (lane % 2) * 16;
    uint32_t a0 = pack4s8(&a_sm[ar][acb + 0]);
    uint32_t a1 = pack4s8(&a_sm[ar][acb + 4]);
    uint32_t a2 = pack4s8(&a_sm[ar][acb + 8]);
    uint32_t a3 = pack4s8(&a_sm[ar][acb + 12]);

    // B: col=n%4 (row of b_sm), k_base=(n/4)*8
    int bc = lane % 4;
    int bkb = (lane / 4) * 8;
    uint32_t b0 = pack4s8(&b_sm[bc][bkb + 0]);
    uint32_t b1 = pack4s8(&b_sm[bc][bkb + 4]);

    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3, a0,a1,a2,a3, b0,b1, 0,0,0,0);

    // C: row=n/2, col_base=(n%2)*4
    int cr = lane / 2;
    int ccb = (lane % 2) * 4;
    C_out[cr*8+ccb+0]=c0; C_out[cr*8+ccb+1]=c1;
    C_out[cr*8+ccb+2]=c2; C_out[cr*8+ccb+3]=c3;
}

// Layout 5: A row=(n/4)%8, col=(n%4)*8, +8 for regs 2,3; B col=(n/4)/2, k=(n%4)*8
// C row=(n/4)%8, col=((n/4)/2)*2+(n%4)*0..1, row+8 for regs 2,3
__global__ void test_layout5(int32_t* C_out,
    const int8_t* A_in, const int8_t* Bt_in)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];

    int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 512; i += 32) ((int8_t*)a_sm)[i] = A_in[i];
    for (int i = threadIdx.x; i < 256; i += 32) ((int8_t*)b_sm)[i] = Bt_in[i];
    __syncthreads();

    // A: row=(n/4), col=(n%4)*8, row+8 for regs 2,3
    int ar1 = lane / 4;
    int ar2 = ar1 + 8;
    int acb = (lane % 4) * 8;
    uint32_t a0 = pack4s8(&a_sm[ar1][acb + 0]);
    uint32_t a1 = pack4s8(&a_sm[ar1][acb + 4]);
    uint32_t a2 = pack4s8(&a_sm[ar2][acb + 0]);
    uint32_t a3 = pack4s8(&a_sm[ar2][acb + 4]);

    // B: col=(n/4), k=(n%4)*8
    // Wait, n/4 for n=0..31 = 0..7 (8 values)
    // n%4 for n=0..31 = 0..3
    // b[0]: B[(n%4)*8+0..3, n/4] = b_sm[n/4][(n%4)*8+0..3]
    int bc = lane / 4;
    int bkb = (lane % 4) * 8;
    uint32_t b0 = pack4s8(&b_sm[bc][bkb + 0]);
    uint32_t b1 = pack4s8(&b_sm[bc][bkb + 4]);

    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3, a0,a1,a2,a3, b0,b1, 0,0,0,0);

    // C: row=(n/4), col=(n%4)*2+{0,1}, row+8 for regs 2,3
    int cr1 = lane / 4;
    int cr2 = cr1 + 8;
    int cc1 = (lane % 4) * 2;
    int cc2 = (lane % 4) * 2 + 1;
    C_out[cr1*8+cc1]=c0; C_out[cr1*8+cc2]=c1;
    C_out[cr2*8+cc1]=c2; C_out[cr2*8+cc2]=c3;
}

static int count_mm(const int32_t* ref, const int32_t* gpu, int n) {
    int e=0; for(int i=0;i<n;i++) if(ref[i]!=gpu[i]) e++; return e;
}

int main() {
    const int M=16, N=8, K=32;
    int8_t h_A[512], h_Bt[256];
    for (int i=0;i<M*K;i++) h_A[i]=(int8_t)((i*3+1)%11-5);
    for (int i=0;i<N*K;i++) h_Bt[i]=(int8_t)((i*7+3)%9-4);

    int32_t h_Cref[128]; memset(h_Cref,0,sizeof(h_Cref));
    for (int i=0;i<M;i++)
        for (int j=0;j<N;j++)
            for (int k=0;k<K;k++)
                h_Cref[i*N+j]+=(int32_t)h_A[i*K+k]*(int32_t)h_Bt[j*K+k];

    int8_t *d_A,*d_Bt; int32_t *d_C;
    cudaMalloc(&d_A,512); cudaMalloc(&d_Bt,256); cudaMalloc(&d_C,512);
    cudaMemcpy(d_A,h_A,512,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bt,h_Bt,256,cudaMemcpyHostToDevice);

    int32_t h_C[128];
    printf("IMMA m16n8k32 layout test (manual pack4s8)\n");
    printf("==========================================\n");

    // Layout 3
    test_layout3<<<1,32>>>(d_C,d_A,d_Bt);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C,d_C,512,cudaMemcpyDeviceToHost);
    int e3=count_mm(h_Cref,h_C,128);
    printf("Layout 3 (A:row=n%%16,col=(n/16)*16; B:col=n%%8,k=(n/8)*8): %d/128 miss %s\n",
           e3, e3==0?"OK!":"WRONG");

    // Layout 4
    test_layout4<<<1,32>>>(d_C,d_A,d_Bt);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C,d_C,512,cudaMemcpyDeviceToHost);
    int e4=count_mm(h_Cref,h_C,128);
    printf("Layout 4 (A:row=n/2,col=(n%%2)*16; B:col=n%%4,k=(n/4)*8): %d/128 miss %s\n",
           e4, e4==0?"OK!":"WRONG");

    // Layout 5 (original)
    test_layout5<<<1,32>>>(d_C,d_A,d_Bt);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C,d_C,512,cudaMemcpyDeviceToHost);
    int e5=count_mm(h_Cref,h_C,128);
    printf("Layout 5 (A:row=n/4,col=(n%%4)*8; B:col=n/4,k=(n%%4)*8): %d/128 miss %s\n",
           e5, e5==0?"OK!":"WRONG");

    // Print first test result for visual comparison
    printf("\nCPU ref:\n");
    for(int i=0;i<4;i++){printf("  ");for(int j=0;j<N;j++)printf("%6d",h_Cref[i*N+j]);printf("\n");}

    cudaFree(d_A); cudaFree(d_Bt); cudaFree(d_C);
    return (e3==0||e4==0||e5==0)?0:1;
}
