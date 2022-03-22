// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 32

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  // TODO: See instructions below

  for (long j = 0; j < n; j=j+BLOCK_SIZE) {
    for (long p = 0; p < k; p++) {
      for(long b1 = 0; b1 < BLOCK_SIZE; b1++){
        for (long i = 0; i < m; i=i+BLOCK_SIZE) {
          for(long b2 = 0; b2 < BLOCK_SIZE; b2++){
            double A_ip = a[(i+b2)+p*m];
            double B_pj = b[p+(j+b1)*k];
            double C_ij = c[(i+b2)+(j+b1)*m];
            C_ij = C_ij + A_ip * B_pj;
            c[(i+b2)+(j+b1)*m] = C_ij;
          }
        }
      }
    }
  }
}

void MMult2(long m, long n, long k, double *a, double *b, double *c) {
  // TODO: See instructions below
  #pragma omp parallel for
  for (long j = 0; j < n; j=j+BLOCK_SIZE) {
    for (long p = 0; p < k; p++) {
      for(long b1 = 0; b1 < BLOCK_SIZE; b1++){
        for (long i = 0; i < m; i=i+BLOCK_SIZE) {
          for(long b2 = 0; b2 < BLOCK_SIZE; b2++){
            double A_ip = a[(i+b2)+p*m];
            double B_pj = b[p+(j+b1)*k];
            double C_ij = c[(i+b2)+(j+b1)*m];
            C_ij = C_ij + A_ip * B_pj;
            c[(i+b2)+(j+b1)*m] = C_ij;
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf("Type        Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_block = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_omp = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c_block[i] = 0;
    for (long i = 0; i < m*n; i++) c_omp[i] = 0;

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }
    double time_ref = t.toc();
  

    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c_block);
    }
    double time_block = t.toc();

    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult2(m, n, k, a, b, c_omp);
    }
    double time_omp = t.toc();
    
    double flops_ref = 2 * m * n * k * NREPEATS / 1e9 / time_ref; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth_ref = 2 * m * n * (k + 1) * NREPEATS * sizeof(double) / 1e9 / time_ref; // TODO: calculate from m, n, k, NREPEATS, time
    
    double flops_block = 2 * m * n * k * NREPEATS / 1e9 / time_block; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth_block = 2 * m * n * (k + 1) * NREPEATS * sizeof(double) / 1e9 / time_block; // TODO: calculate from m, n, k, NREPEATS, time

    double flops_omp = 2 * m * n * k * NREPEATS / 1e9 / time_omp; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth_omp = 2 * m * n * (k + 1) * NREPEATS * sizeof(double) / 1e9 / time_omp; // TODO: calculate from m, n, k, NREPEATS, time

    double max_err_ref = 0;
    double max_err_block = 0;
    double max_err_omp = 0;

    for (long i = 0; i < m*n; i++) max_err_block = std::max(max_err_block, fabs(c_block[i] - c_ref[i]));
    for (long i = 0; i < m*n; i++) max_err_omp = std::max(max_err_omp, fabs(c_omp[i] - c_ref[i]));

    printf("Reference %10d %10f %10f %10f %10e\n", p, time_ref, flops_ref, bandwidth_ref, max_err_ref);
    printf("Blockwise %10d %10f %10f %10f %10e\n", p, time_block, flops_block, bandwidth_block, max_err_block);
    printf("Block+OMP %10d %10f %10f %10f %10e\n", p, time_omp, flops_omp, bandwidth_omp, max_err_omp);
    printf("------------------------------------------------------------------------------------------\n");

    aligned_free(a);
    aligned_free(b);
    aligned_free(c_block);
    aligned_free(c_omp);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
