#include <stdio.h>
#include <math.h>
#include "utils.h"

double calculateNorm(long N, double* A, double* u, double* f){

    double* p = (double*) malloc(N * sizeof(double));

    for(int i=0; i<N; i++){
        p[i] = 0;
    }

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            p[i] = p[i] + (A[j+i*N]*u[j])/pow((N+1), 2);
        }
    }

    double residue = 0;
    
    for(int i=0; i<N; i++){
        residue = residue + pow((p[i] - f[i]), 2);
    }

    double norm = sqrt(residue);
    free(p);

    return norm;
}

double* calculateU(long N, double* A, double* u, double* f){

    double* newU = (double*) malloc(N * sizeof(double));

    for(int i=0; i<N; i++){
        double summation = 0;
        for(int j=0; j<N; j++){
            if(j<i){
                summation = summation + (A[j+i*N]*newU[j]/pow((N+1),2));
            }
            else if(j>i){
                summation = summation + (A[j+i*N]*u[j]/pow((N+1),2));
            }
        }
        newU[i] = (f[i] - summation)*pow((N+1), 2)/A[i+i*N];
    }

    return newU;
}

int main(int argc, char** argv){

    long N = read_option<long>("-n", argc, argv);
    double h = 1/(N+1);

    Timer t;
    t.tic();

    double* A = (double*) malloc(N * N * sizeof(double));
    double* u = (double*) malloc(N * sizeof(double));
    double* f = (double*) malloc(N * sizeof(double));

    for(int i=0; i< N; i++){
        f[i] = 1;
        u[i] = 0;
    }

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(j==i)
                A[j+i*N] = 2;
            else if(j==i+1 || i==j+1)
                A[j+i*N] = -1;
            else
                A[j+i*N] = 0;
        }
    }

    int iteration = 1;
    double initial_norm = calculateNorm(N, A, u, f);

    printf("Iteration: %d | Norm: %f\n", iteration, initial_norm);

    double threshold = initial_norm / 1e6;
    double norm = initial_norm;

    while(norm >= threshold && iteration < 5000){
        
        iteration++;
        u = calculateU(N, A, u, f);
        norm = calculateNorm(N, A, u, f);
        printf("Iteration: %d | Norm: %f\n", iteration, norm);
        if(iteration == 100)
            printf("Time taken for 100 iterations: %10f\n", t.toc());
    }
    
    // printf("u: ");

    // for(int i=0; i<N; i++)
    //     printf("%f ", u[i]);

    free(A);
    free(u);
    free(f);
}