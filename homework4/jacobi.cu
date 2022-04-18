#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <stdlib.h>

void jac2d_serial(double *newU, double *f, double *u, long N, double h)
{
    for (long i = 1; i < N; i++)
    {
        for (long j = i * N; j < (i + 1) * N - 1; j++)
        {
            if (j % N > 0 && j % N < N - 1)
                newU[j] = (h * h * f[j] + u[j - 1] + u[j + 1] + u[j - N] + u[j + N]) * 0.25;   
        }
    }
}



void Check_CUDA_Error(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

#define BLOCK_SIZE 1024


__global__ void jacobi_2d(double *newU, double *f, double *u, long N, double h)
{
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N && idx < N * (N - 1) && idx % N > 0 && idx % N < N - 1)
        newU[idx] = (u[idx - 1] + u[idx + 1] + u[idx - N] + u[idx + N] + h * h * f[idx]) * 0.25;
}

int main()
{
    long N = (1UL<<7);
    double h = 1.0 / (N + 1.0);

    double *u = (double*) malloc(N * N * sizeof(double));
    double *f = (double*) malloc(N * N * sizeof(double));
    double *newU_cpu = (double*) malloc(N * N * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N * N; i++){
        u[i] = 0.0;
        f[i] = 1.0;
        newU_cpu[i] = 0.0;
    }

    double *newU_gpu_d, *u_d, *f_d;
    cudaMalloc(&newU_gpu_d, N * N * sizeof(double));
    cudaMalloc(&u_d, N * N * sizeof(double));
    cudaMalloc(&f_d, N * N * sizeof(double));

    cudaMemcpy(newU_gpu_d, u, N * N * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(u_d, u, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_d, f, N * N * sizeof(double), cudaMemcpyHostToDevice);


    double cpuStartTime = omp_get_wtime();

    for (long iteration = 0; iteration < 1000; iteration++)
    {
        jac2d_serial(newU_cpu, f, u, N, h);
        for (long i = 1; i < N * N; i++) 
            u[i] = newU_cpu[i];
        //u = newU_cpu;
    }
    
    printf("CPU Time = %f s\n", (omp_get_wtime()-cpuStartTime));


    double *u_gpu = (double*) malloc(N * N * sizeof(double));
   
    double gpuStartTime = omp_get_wtime();
    for (long iteration = 0; iteration < 1000; iteration++)
    {
        jacobi_2d<<<N, BLOCK_SIZE>>>(newU_gpu_d, f_d, u_d, N, h);
        u_d = newU_gpu_d;
    }
    cudaMemcpy(u_gpu, u_d, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("GPU Time = %f s\n", (omp_get_wtime() - gpuStartTime));
    
    
    double err = 0;
    #pragma omp parallel for reduction(+:err)
    for (long i = 0; i < N*N; i++) 
        err = err + (u_gpu[i] - u[i]) * (u_gpu[i] - u[i]);
    printf("Error = %f\n", err);


    cudaFree(newU_gpu_d);
    cudaFree(u_d);
    cudaFree(f_d);
    free(f);
    free(u_gpu);
    free(newU_cpu);
    free(u);
    return 0;
}