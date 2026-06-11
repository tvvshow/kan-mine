// imma_interleave.cu — Test INTERLEAVED K-dimension layout.
// Hypothesis from CUTLASS crosswise pattern:
// A: row=n%16, k_base=(n/16)*4, STRIDE=8 between registers
//   a[0]=cols (n/16)*4+0..3, a[1]=cols (n/16)*4+8..11,
//   a[2]=cols (n/16)*4+16..19, a[3]=cols (n/16)*4+24..27
// B: col=n%8, k_base=(n/8)*8, NO stride (consecutive)
//   b[0]=rows (n/8)*8+0..3, b[1]=rows (n/8)*8+4..7

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

// Test: dump raw IMMA output, CPU finds mapping
__global__ void test_interleave(int32_t* C_raw,
    const int8_t* A_in, const int8_t* Bt_in, int variant)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];
    int lane=threadIdx.x&31;
    for(int i=lane;i<512;i+=32)((int8_t*)a_sm)[i]=A_in[i];
    for(int i=lane;i<256;i+=32)((int8_t*)b_sm)[i]=Bt_in[i];
    __syncthreads();

    uint32_t a0,a1,a2,a3,b0,b1;

    if (variant == 0) {
        // Interleaved A: row=n%16, k_base=(n/16)*4, stride 8 between regs
        int ar = lane % 16;
        int kb = (lane / 16) * 4;
        a0 = pack4s8(&a_sm[ar][kb + 0]);
        a1 = pack4s8(&a_sm[ar][kb + 8]);
        a2 = pack4s8(&a_sm[ar][kb + 16]);
        a3 = pack4s8(&a_sm[ar][kb + 24]);
        // B: col=n%8, k_base=(n/8)*8
        int bc = lane % 8;
        int bkb = (lane / 8) * 8;
        b0 = pack4s8(&b_sm[bc][bkb + 0]);
        b1 = pack4s8(&b_sm[bc][bkb + 4]);
    } else if (variant == 1) {
        // Same as 0 but B also interleaved
        int ar = lane % 16;
        int kb = (lane / 16) * 4;
        a0 = pack4s8(&a_sm[ar][kb + 0]);
        a1 = pack4s8(&a_sm[ar][kb + 8]);
        a2 = pack4s8(&a_sm[ar][kb + 16]);
        a3 = pack4s8(&a_sm[ar][kb + 24]);
        // B interleaved: col=n%4, k_base=(n/4)*4, stride 8 between regs
        int bc = lane % 4;
        int bkb = (lane / 4) * 4;
        b0 = pack4s8(&b_sm[bc][bkb + 0]);
        b1 = pack4s8(&b_sm[bc][bkb + 8]);
    } else if (variant == 2) {
        // Interleaved A with stride 16
        int ar = lane % 16;
        int kb = (lane / 16) * 4;
        a0 = pack4s8(&a_sm[ar][kb + 0]);
        a1 = pack4s8(&a_sm[ar][kb + 4]);
        a2 = pack4s8(&a_sm[ar][kb + 16]);  // stride 16, not 8
        a3 = pack4s8(&a_sm[ar][kb + 20]);
        int bc = lane % 8;
        int bkb = (lane / 8) * 8;
        b0 = pack4s8(&b_sm[bc][bkb + 0]);
        b1 = pack4s8(&b_sm[bc][bkb + 4]);
    } else {
        // All interleaved with stride 8, B col=n%4 kbase=(n%4)*4+(n/4)/2*16
        int ar = lane % 16;
        int kb = (lane / 16) * 4;
        a0 = pack4s8(&a_sm[ar][kb + 0]);
        a1 = pack4s8(&a_sm[ar][kb + 8]);
        a2 = pack4s8(&a_sm[ar][kb + 16]);
        a3 = pack4s8(&a_sm[ar][kb + 24]);
        // B: col = n/8, k = (n%8)*4 interleaved?
        // 32 threads: col=n/8 (0..3), k=(n%8)*4
        // That gives 4 cols x 8 k-groups = 32 cols x 32 k = 1024, but B is 32x8
        int bc = lane / 8;
        int bkb = (lane % 8) * 4;
        b0 = pack4s8(&b_sm[bc][bkb + 0]);
        b1 = pack4s8(&b_sm[bc][bkb + 8]);
    }

    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1,0,0,0,0);
    C_raw[lane*4+0]=c0; C_raw[lane*4+1]=c1;
    C_raw[lane*4+2]=c2; C_raw[lane*4+3]=c3;
}

int main() {
    const int M=16,N=8,K=32;
    int8_t h_A[512],h_Bt[256];
    for(int i=0;i<M*K;i++) h_A[i]=(int8_t)((i*32+i)%127+1);
    for(int i=0;i<N*K;i++) h_Bt[i]=(int8_t)((i*8+i)%127+1);

    int32_t h_Cref[128]; memset(h_Cref,0,sizeof(h_Cref));
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<K;k++)
                h_Cref[i*N+j]+=(int32_t)h_A[i*K+k]*(int32_t)h_Bt[j*K+k];

    int8_t *d_A,*d_Bt; int32_t *d_C;
    cudaMalloc(&d_A,512); cudaMalloc(&d_Bt,256); cudaMalloc(&d_C,512);
    cudaMemcpy(d_A,h_A,512,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bt,h_Bt,256,cudaMemcpyHostToDevice);

    const char* names[] = {"A-inter B-plain","A-inter B-inter","A-stride16 B-plain","A-inter B-alt"};
    for (int v = 0; v < 4; v++) {
        test_interleave<<<1,32>>>(d_C,d_A,d_Bt,v);
        cudaDeviceSynchronize();
        int32_t h_C[128];
        cudaMemcpy(h_C,d_C,512,cudaMemcpyDeviceToHost);

        int found=0;
        for(int t=0;t<128;t++)
            for(int r=0;r<128;r++)
                if(h_C[t]==h_Cref[r]){found++;break;}

        // Also try all (row,col) C mappings and count matches
        // For each thread, try C mapped as: row=n%16, col=(n/16)*4+reg
        int mapped=0;
        for(int n=0;n<32;n++){
            int cr=n%16, ccb=(n/16)*4;
            for(int r=0;r<4;r++){
                if(h_C[n*4+r]==h_Cref[cr*8+ccb+r]) mapped++;
            }
        }
        // Also try: row=n%16, col=(n/16)*4, regs = 4 consecutive
        // And: row=n/4, col=(n%4)*2+{0,1}, row+8
        int mapped2=0;
        for(int n=0;n<32;n++){
            int cr1=n/4,cr2=cr1+8,cc1=(n%4)*2,cc2=cc1+1;
            if(h_C[n*4+0]==h_Cref[cr1*8+cc1]) mapped2++;
            if(h_C[n*4+1]==h_Cref[cr1*8+cc2]) mapped2++;
            if(h_C[n*4+2]==h_Cref[cr2*8+cc1]) mapped2++;
            if(h_C[n*4+3]==h_Cref[cr2*8+cc2]) mapped2++;
        }
        printf("V%d (%s): raw_match=%d/128  C_map1=%d/128  C_map2=%d/128\n",
               v, names[v], found, mapped, mapped2);
    }

    // Print CPU ref
    printf("\nCPU ref C[16][8] (first 4 rows):\n");
    for(int i=0;i<4;i++){
        printf("  ");for(int j=0;j<N;j++)printf("%8d",h_Cref[i*N+j]);printf("\n");
    }

    cudaFree(d_A);cudaFree(d_Bt);cudaFree(d_C);
    return 0;
}
