#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define BLOCK_SIZE 1024

void matVecMult_non_cuda(double* M, double* x, double* result, long N){

    for(long i=0; i<N; i++){
        for(long j=0; j<N; j++)
            result[i] = result[i] + M[i*N+j] * x[j];
    }
}

__global__ void matVec_product_kernel(const double *M, const double *x, double *prod, long N) {
    __shared__ double sharedMem[BLOCK_SIZE];
    
    sharedMem[threadIdx.x] = 0;
    for(long i=threadIdx.x; i<N; i=i+BLOCK_SIZE){
        sharedMem[threadIdx.x] += M[blockIdx.x * N + i] * x[i];
    }

    __syncthreads();
    if(threadIdx.x < 512) 
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 512];
    __syncthreads();
    if(threadIdx.x < 256) 
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 256];
    __syncthreads();
    if(threadIdx.x < 128) 
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 128];
    __syncthreads();
    if(threadIdx.x <  64) 
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 64];
    __syncthreads();
    if(threadIdx.x <  32){
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 32];
        __syncwarp();
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 16];
        __syncwarp();
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 8];
        __syncwarp();
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 4];
        __syncwarp();
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + 2];
        __syncwarp();
        if(threadIdx.x == 0) 
            prod[blockIdx.x] = sharedMem[0] + sharedMem[1];
    }
}

void matVecMult_gpu(double* M, double* x, double* result, long N){

    double *M_d, *x_d, *prod_d;
    cudaMalloc(&M_d, N * N * sizeof(double));
    cudaMalloc(&x_d, N * sizeof(double));

    cudaMemcpy(M_d, M, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&prod_d, N * sizeof(double));

    double gpuStartTime = omp_get_wtime();

    matVec_product_kernel<<<N, BLOCK_SIZE>>>(M_d, x_d, prod_d, N);
    cudaMemcpy(result, prod_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("GPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime() - gpuStartTime)/1e9);

    cudaFree(x_d);
    cudaFree(M_d);
    cudaFree(prod_d);
}

int main(){
    
    long N = (1UL<<10);

    double *M, *x, *result, *result1;

    M = (double*) malloc(N * N * sizeof(double));
    x = (double*) malloc(N * sizeof(double));
    result = (double*) malloc(N * sizeof(double));
    result1 = (double*) malloc(N * sizeof(double));


    #pragma omp parallel for
    for(long i=0; i<N*N; i++){
        M[i] = drand48();
    }

    #pragma omp parallel for
    for(long i=0; i<N; i++){
        x[i] = drand48();
    }

    double cpuStartTime = omp_get_wtime();
    matVecMult_non_cuda(M, x, result, N);
    printf("CPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-cpuStartTime)/1e9);
    
    matVecMult_gpu(M, x, result1, N);

    double err = 0;
    #pragma omp parallel for reduction(+:err)
    for (long i = 0; i < N; i++) 
        err = err + (result1[i] - result[i]) * (result1[i] - result[i]);
    
    printf("Matrix Vector product error = %f\n\n", err);
    
    free(x);
    free(M);
    free(result);
    free(result1);

    return 0;
}