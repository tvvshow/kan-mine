// imma_map.cu — Determine C fragment mapping using structured inputs.
// A[i][k] = (i+1) for all k  => each row has distinct value
// B[k][j] = (j+1) for all k  => each column has distinct value
// C[i][j] = sum_k (i+1)*(j+1) = 32*(i+1)*(j+1)
// Each (i,j) pair gives a UNIQUE C value => can map thread->(row,col)

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

// Load A fragments in Layout 3 (row=n%16, col=(n/16)*16)
// Load B fragments in Layout 3 (col=n%8, k=(n/8)*8)
__global__ void map_test(int32_t* C_raw, int32_t* C_mapped,
    const int8_t* A_in, const int8_t* Bt_in)
{
    __shared__ int8_t a_sm[16][32];
    __shared__ int8_t b_sm[8][32];
    int lane=threadIdx.x&31;
    for(int i=lane;i<512;i+=32)((int8_t*)a_sm)[i]=A_in[i];
    for(int i=lane;i<256;i+=32)((int8_t*)b_sm)[i]=Bt_in[i];
    __syncthreads();

    // A: row=lane%16, col=(lane/16)*16 + reg*4
    int ar=lane%16, acb=(lane/16)*16;
    uint32_t a0=pack4s8(&a_sm[ar][acb+0]);
    uint32_t a1=pack4s8(&a_sm[ar][acb+4]);
    uint32_t a2=pack4s8(&a_sm[ar][acb+8]);
    uint32_t a3=pack4s8(&a_sm[ar][acb+12]);

    // B: col=lane%8, k=(lane/8)*8 + reg*4
    int bc=lane%8, bkb=(lane/8)*8;
    uint32_t b0=pack4s8(&b_sm[bc][bkb+0]);
    uint32_t b1=pack4s8(&b_sm[bc][bkb+4]);

    int c0=0,c1=0,c2=0,c3=0;
    imma16832(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1,0,0,0,0);

    // Raw dump
    C_raw[lane*4+0]=c0; C_raw[lane*4+1]=c1;
    C_raw[lane*4+2]=c2; C_raw[lane*4+3]=c3;

    // Try all C mappings and write best one
    // Mapping 1: row=lane%16, col=(lane/16)*4+reg
    // Mapping 2: row=lane/4, col=(lane%4)*2+{0,1}, row+8 for reg2,3
    // We'll just write raw for now; CPU will analyze
}

int main() {
    const int M=16,N=8,K=32;
    int8_t h_A[512],h_Bt[256];

    // A[i][k] = (i+1) for all k
    for(int i=0;i<M;i++)
        for(int k=0;k<K;k++)
            h_A[i*K+k]=(int8_t)(i+1);  // values 1..16

    // B^T[j][k] = (j+1) for all k => B[k][j] = (j+1)
    for(int j=0;j<N;j++)
        for(int k=0;k<K;k++)
            h_Bt[j*K+k]=(int8_t)(j+1);  // values 1..8

    // CPU ref: C[i][j] = 32*(i+1)*(j+1)
    int32_t h_Cref[128];
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
            h_Cref[i*N+j]=32*(i+1)*(j+1);

    int8_t *d_A,*d_Bt; int32_t *d_C,*d_Cm;
    cudaMalloc(&d_A,512); cudaMalloc(&d_Bt,256);
    cudaMalloc(&d_C,512); cudaMalloc(&d_Cm,512);
    cudaMemcpy(d_A,h_A,512,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bt,h_Bt,256,cudaMemcpyHostToDevice);

    map_test<<<1,32>>>(d_C,d_Cm,d_A,d_Bt);
    cudaDeviceSynchronize();

    int32_t h_C[128];
    cudaMemcpy(h_C,d_C,512,cudaMemcpyDeviceToHost);

    printf("IMMA C fragment mapping (A[i][k]=i+1, B[k][j]=j+1)\n");
    printf("Expected: C[i][j] = 32*(i+1)*(j+1)\n");
    printf("================================================\n\n");

    // CPU ref table
    printf("CPU ref C[16][8]:\n   ");
    for(int j=0;j<N;j++) printf("%7d",j);
    printf("\n");
    for(int i=0;i<M;i++){
        printf("%2d ",i);
        for(int j=0;j<N;j++) printf("%7d",h_Cref[i*N+j]);
        printf("\n");
    }

    // For each thread, find which C[i][j] matches each of its 4 values
    printf("\nThread -> C mapping:\n");
    int total_match=0;
    for(int t=0;t<32;t++){
        printf("T%2d:",t);
        for(int r=0;r<4;r++){
            int gv=h_C[t*4+r];
            int found=0;
            for(int i=0;i<M&&!found;i++)
                for(int j=0;j<N&&!found;j++)
                    if(gv==h_Cref[i*N+j]){
                        printf(" [%d,%d]=%d",i,j,gv);
                        found=1; total_match++;
                    }
            if(!found) printf(" ???=%d",gv);
        }
        printf("\n");
    }
    printf("\nTotal matches: %d/128\n", total_match);

    cudaFree(d_A);cudaFree(d_Bt);cudaFree(d_C);cudaFree(d_Cm);
    return 0;
}
