#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"

double calculateNorm(long N, double* u, double* f){

    double h = 1./(N+1);
    double negDeltaU;

    double residue = 0;
    double diff;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:residue)
    #endif
    for(int i=1; i<=N; i++){
        for(int j=1; j<=N; j++){
            negDeltaU = (-u[(i-1)*(N+2)+j] - u[i*(N+2)+j-1] + 4*u[i*(N+2)+j] - u[(i+1)*(N+2)+j] - u[i*(N+2)+j+1])/(h*h);
            diff = f[(i-1)*N+(j-1)] - negDeltaU;
            residue += diff*diff;
        }
    }

    double norm = sqrt(residue);
   
    return norm;
}

void calculateU(long N, double* u, double* newU, double* f){

    double h = 1./(N+1);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int i=1; i<=N; i++){
        for(int j=1; j<=N; j++){
            newU[i*(N+2)+j] = 0.25 * (h*h*f[(i-1)*N+j-1] + u[(i-1)*(N+2)+j] + u[i*(N+2)+j-1] + u[(i+1)*(N+2)+j] + u[i*(N+2)+j+1]);
        }
    }
}

int main(int argc, char** argv){

    printf(" Size   Iterations    Time \n");
    
    for(int N=100; N<=1000; N=N+100){

        Timer t;
        t.tic();

        double* u = (double*) malloc((N+2) * (N+2) * sizeof(double));
        double* newU = (double*) malloc((N+2) * (N+2) * sizeof(double));
        double* f = (double*) malloc(N * N * sizeof(double));

        for(int i = 0; i < N+2; i++){
            for(int j = 0; j < N+2; j++){
                if(i<N && j<N)
                    f[i*N+j] = 1;
                u[i*N+j] = 0;
                newU[i*N+j] = 0;
            }
        }

        int iteration = 1;
        double initial_norm = calculateNorm(N, u, f);

        

        double threshold = initial_norm / 1e6;
        double norm = initial_norm;

        while(norm >= threshold && iteration < 5000){
        
            iteration++;
            calculateU(N, u, newU, f);
            u = newU;

            norm = calculateNorm(N, u, f);
            //printf("Iteration: %d | Norm: %f\n", iteration, norm);
        }
        
        printf("%5d %10d %10f\n", N, iteration, t.toc());
        free(u);
        free(f);
    }
}