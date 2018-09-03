#ifndef MAIN_HPP
#define MAIN_HPP

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <algorithm>
#include <list>
#include <vector>
#include <mm_malloc.h>
#include <time.h>
#include <functional>
#include <omp.h>
#include <mkl.h>

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"

#define GAUSS_SEIDEL 0
#define JACOBI 1
#define IBLOCK_JACOBI_GS 2

typedef struct{
   int *smooth_relax;
   int *smooth_sweeps;
   int *cycles;
   double *smooth_wtime;
   double *residual_wtime;
   double *restrict_wtime;
   double *prolong_wtime;
   double solve_wtime;
   double r_norm2;
   double r0_norm2;
}OutputData;

typedef struct{
   int num_pre_smooth_sweeps;
   int num_post_smooth_sweeps;
   int num_fine_smooth_sweeps;
   int num_coarse_smooth_sweeps;
   int num_cycles;
   double tol;
   int format_output_flag;
   int print_reshist_flag;
   int num_threads;
   double smooth_weight;
   int smoother;
}InputData;

typedef struct{
   HYPRE_Real **f;
   HYPRE_Real **u;
   HYPRE_Real **u_prev;
   HYPRE_Real **u_fine;
   HYPRE_Real **u_fine_prev;
   HYPRE_Real **u_coarse;
   HYPRE_Real **u_coarse_prev;
   HYPRE_Real **y;
   HYPRE_Real **r;
   HYPRE_Real **r_fine;
   HYPRE_Real **r_coarse;
   HYPRE_Real **e;
}VectorData;

typedef struct{
   double *a;
   MKL_INT *ia;
   MKL_INT *ja;
   MKL_INT n;
   MKL_INT nnz;
}PardisoCSR;

typedef struct{
   double wtime_setup;
   void *pt[64];
   MKL_INT maxfct;
   MKL_INT mnum;
   MKL_INT mtype;
   MKL_INT phase;
   MKL_INT *perm;
   MKL_INT nrhs;
   MKL_INT iparm[64];
   MKL_INT msglvl;
   MKL_INT error;
   MKL_INT ddum;
   MKL_INT idum;
   PardisoCSR csr;
}PardisoInfo;

typedef struct{
   PardisoCSR csr;
   PardisoInfo info;
}PardisoData;

typedef struct{
   hypre_CSRMatrix **A;
   hypre_CSRMatrix **P;
   hypre_CSRMatrix **R;
}MatrixData;

typedef struct{
   int *thread_lev;
   int *n;
}ThreadData;

typedef struct{
   int num_levels;
   int *n;
}GridData;

typedef struct{
   ThreadData thread;
   MatrixData data;
   VectorData vector;
   MatrixData matrix;
   InputData input;
   OutputData output;
   GridData grid;
   PardisoData pardiso;
}AllData;

#endif
