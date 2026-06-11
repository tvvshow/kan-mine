// imma_simple.cu — Simplest possible IMMA test.
// A = all 1s, B = all 1s => C[i][j] = sum(1*1, k=0..31) = 32 everywhere.
// If IMMA outputs 32 for all elements, the instruction works and only the
// fragment layout needs fixing.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

static __device__ __forceinline__ uint32_t pack4s8(const int8_t* p) {
    return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1]<<8)
         | ((uint32_t)(uint8_t)p[2]<<16) | ((uint32_t)(uint8_t)p[3]<<24);
}
static __device__ void imma16832(
    int &d0,int &d1,int &d2,int &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1,
    int c0,int c1,int c2,int c3)
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n\t"
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "r"(c0),"r"(c1),"r"(c2),"r"(c3));
}

__global__ void simple_test(int32_t* C_raw, int variant)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];
    int lane=threadIdx.x&31;

    // Fill A and B with all 1s
    for(int i=lane;i<512;i+=32) ((int8_t*)a_sm)[i]=1;
    for(int i=lane;i<256;i+=32) ((int8_t*)b_sm)[i]=1;
    __syncthreads();

    uint32_t a0,a1,a2,a3,b0,b1;

    if (variant == 0) {
        // All ones packed: each reg = 0x01010101
        a0=a1=a2=a3=0x01010101u;
        b0=b1=0x01010101u;
    } else if (variant == 1) {
        // Load from smem with Layout 3
        int ar=lane%16, acb=(lane/16)*16;
        a0=pack4s8(&a_sm[ar][acb+0]); a1=pack4s8(&a_sm[ar][acb+4]);
        a2=pack4s8(&a_sm[ar][acb+8]); a3=pack4s8(&a_sm[ar][acb+12]);
        int bc=lane%8, bkb=(lane/8)*8;
        b0=pack4s8(&b_sm[bc][bkb+0]); b1=pack4s8(&b_sm[bc][bkb+4]);
    } else if (variant == 2) {
        // Load with Layout 5
        int ar1=lane/4, ar2=ar1+8, acb=(lane%4)*8;
        a0=pack4s8(&a_sm[ar1][acb+0]); a1=pack4s8(&a_sm[ar1][acb+4]);
        a2=pack4s8(&a_sm[ar2][acb+0]); a3=pack4s8(&a_sm[ar2][acb+4]);
        int bc=lane/4, bkb=(lane%4)*8;
        b0=pack4s8(&b_sm[bc][bkb+0]); b1=pack4s8(&b_sm[bc][bkb+4]);
    } else {
        // Interleaved A
        int ar=lane%16, kb=(lane/16)*4;
        a0=pack4s8(&a_sm[ar][kb+0]); a1=pack4s8(&a_sm[ar][kb+8]);
        a2=pack4s8(&a_sm[ar][kb+16]); a3=pack4s8(&a_sm[ar][kb+24]);
        int bc=lane%8, bkb=(lane/8)*8;
        b0=pack4s8(&b_sm[bc][bkb+0]); b1=pack4s8(&b_sm[bc][bkb+4]);
    }

    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1,0,0,0,0);
    C_raw[lane*4+0]=c0; C_raw[lane*4+1]=c1;
    C_raw[lane*4+2]=c2; C_raw[lane*4+3]=c3;
}

int main() {
    int32_t *d_C;
    cudaMalloc(&d_C, 128*4);

    printf("IMMA simple test: A=1, B=1 => C=32 everywhere\n");
    printf("================================================\n");

    const char* names[] = {"literal 0x01010101","Layout3","Layout5","Interleaved"};
    for (int v = 0; v < 4; v++) {
        simple_test<<<1,32>>>(d_C, v);
        cudaDeviceSynchronize();
        int32_t h_C[128];
        cudaMemcpy(h_C, d_C, 128*4, cudaMemcpyDeviceToHost);

        int all32=1, minv=h_C[0], maxv=h_C[0];
        for(int i=0;i<128;i++){
            if(h_C[i]!=32) all32=0;
            if(h_C[i]<minv)minv=h_C[i];
            if(h_C[i]>maxv)maxv=h_C[i];
        }
        printf("V%d (%s): all_32=%s  range=[%d..%d]\n",
               v, names[v], all32?"YES":"NO", minv, maxv);
        if (!all32) {
            printf("  First 4 threads:\n");
            for(int t=0;t<4;t++)
                printf("    T%2d: %d %d %d %d\n",t,h_C[t*4],h_C[t*4+1],h_C[t*4+2],h_C[t*4+3]);
        }
    }

    // Also test with A=2, B=3 => C=2*3*32=192
    printf("\nBonus: A=2, B=3 => expect 192\n");
    // Manually set fragments to 2 and 3
    // Since all smem has 1s, use literal values
    // pack 4 x s8(2) = 0x02020202, pack 4 x s8(3) = 0x03030303
    // Just test with variant 0 (literal)
    // Can't easily change smem, so test inline:
    // Actually we can test by running a quick kernel variant
    // Let me skip this for now.

    cudaFree(d_C);
    return 0;
}
