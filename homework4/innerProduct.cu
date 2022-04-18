#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define BLOCK_SIZE 1024

void inner_product_non_cuda(double* x, double* y, double* result, long N){

    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(long i=0; i<N; i++)
        sum = sum + x[i]*y[i];

    *result = sum;
}


__global__ void inner_product_kernel(const double *x, const double *y, double *prod, long N) {
    __shared__ double sharedMem[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if(idx < N) 
        sharedMem[threadIdx.x] = x[idx] * y[idx]; // Now multiply a[i] * b[i].
    else 
        sharedMem[threadIdx.x] = 0;

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

__global__ void reduction_kernel(double* prod, const double* partial, long N){
    __shared__ double sharedMem[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if(idx < N) 
        sharedMem[threadIdx.x] = partial[idx];
    else 
        sharedMem[threadIdx.x] = 0;

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
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x +  64];
    __syncthreads();
    if(threadIdx.x <  32){
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x +  32];
        __syncwarp();
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x +  16];
        __syncwarp();
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x +   8];
        __syncwarp();
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x +   4];
        __syncwarp();
        sharedMem[threadIdx.x] += sharedMem[threadIdx.x +   2];
        __syncwarp();
        if(threadIdx.x == 0) 
            prod[blockIdx.x] = sharedMem[0] + sharedMem[1];
    }
}

void inner_product_gpu(double* x, double* y, double* result, long N){

    double *x_d, *y_d;
    cudaMalloc(&x_d, N * sizeof(double));
    cudaMalloc(&y_d, N * sizeof(double));

    cudaMemcpy(x_d, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(double), cudaMemcpyHostToDevice);

    double *prod_d; 
    long N_total = 1; 
    for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) 
        N_total += i;
    cudaMalloc(&prod_d, N_total*sizeof(double)); 

    double gpuStartTime = omp_get_wtime(); 

    double *partialSum_d = prod_d;
    long numberOfBlocks = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);

    inner_product_kernel<<<numberOfBlocks,BLOCK_SIZE>>>(x_d, y_d, partialSum_d, N);

    while (numberOfBlocks > 1) {
        long temp = numberOfBlocks;
        numberOfBlocks = (numberOfBlocks+BLOCK_SIZE-1)/(BLOCK_SIZE);
        reduction_kernel<<<numberOfBlocks,BLOCK_SIZE>>>(partialSum_d + temp, partialSum_d, temp);
        partialSum_d += temp;
    }

    cudaMemcpy(result, partialSum_d, sizeof(double), cudaMemcpyDeviceToHost);
  
    printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime() - gpuStartTime)/1e9);

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(prod_d);
}

int main(){

    long N = (1UL<<25);

    double *x, *y;

    x = (double*) malloc(N * sizeof(double));
    y = (double*) malloc(N * sizeof(double));

    #pragma omp parallel for
    for(long i=0; i<N; i++){
        x[i] = drand48();
        y[i] = drand48();
    }

    double result = 0;
    double cpuStartTime = omp_get_wtime();
    inner_product_non_cuda(x, y, &result, N);
    printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-cpuStartTime)/1e9);

    double result1 = 0;
    inner_product_gpu(x, y, &result1, N);    
    printf("Inner product Error = %f\n\n", fabs(result1-result));

    free(x);
    free(y);
    return 0;

}