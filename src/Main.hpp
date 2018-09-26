#ifndef MAIN_HPP
#define MAIN_HPP

#include <fstream>
#include <iostream>
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

#include "mfem.hpp"

#define JACOBI 0
#define GAUSS_SEIDEL 1
#define HYBRID_JACOBI_GAUSS_SEIDEL 2
#define SYMM_JACOBI 3
#define SEMI_ASYNC_GAUSS_SEIDEL 4
#define ASYNC_GAUSS_SEIDEL 5
#define L1_JACOBI 6

#define MULT 0
#define AFACX 1
#define MULTADD 2
#define ASYNC_AFACX 3
#define ASYNC_MULTADD 4

#define ONE_LEVEL 0
#define ALL_LEVELS 1

#define HALF_THREADS 0
#define EQUAL_THREADS 1
#define BALANCED_THREADS 2

#define FULL_ASYNC 0
#define SEMI_ASYNC 1

#define LAPLACE_2D5PT 0
#define LAPLACE_3D27PT 1
#define MFEM_LAPLACE 2
#define MFEM_ELAST 3

using namespace std;
using namespace mfem;

typedef struct{
   int *smooth_relax;
   int *smooth_sweeps;
   int *cycles;
   double *smooth_wtime;
   double *residual_wtime;
   double *restrict_wtime;
   double *prolong_wtime;
   double setup_wtime;
   double hypre_setup_wtime;
   double prob_setup_wtime;
   double solve_wtime;
   double hypre_solve_wtime;
   double r_norm2;
   double r0_norm2;
   double hypre_e_norm2;
   double mfem_e_norm2;
   int sim_time_instance;
   int sim_cycle_time_instance;
}OutputData;

typedef struct{
   int num_pre_smooth_sweeps;
   int num_post_smooth_sweeps;
   int num_fine_smooth_sweeps;
   int num_coarse_smooth_sweeps;
   int num_cycles;
   int num_threads;
   double tol;
   int format_output_flag;
   int print_reshist_flag;
   int print_output_flag;
   int check_resnorm_flag;
   int global_conv_flag;
   double smooth_weight;
   int smoother;
   int solver;
   int async_flag;
   int async_type;
   int thread_part_type;
   int thread_part_distr_type;
   int converge_test_type;
   int test_problem;
   int hypre_test_error_flag;
   int mfem_test_error_flag;
   int mfem_solve_print_flag;
   int sim_grid_wait;
   int sim_read_delay;
   int print_grid_wait_flag;
   int print_level_stats_flag;
   int smooth_interp_type;
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
   int zero_flag;
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
   int ref_levels;
   int order;
   char mesh_file[1000];
   double *u;
}MfemData;

typedef struct{
   hypre_CSRMatrix **A;
   hypre_CSRMatrix **P;
   hypre_CSRMatrix **R;
   double **L1_row_norm;
}MatrixData;

typedef struct{
   vector<vector<int>> thread_levels;
   vector<vector<int>> level_threads;
   omp_lock_t lock;
   int *barrier_root;
   int **barrier_flags;
   double *loc_sum;
   int **A_ns;
   int **A_ne;
   int **R_ns;
   int **R_ne;
   int **P_ns;
   int **P_ne;
   int converge_flag;
}ThreadData;

typedef struct{
   int num_levels;
   int *n;
   int global_num_correct;
   int global_cycle_num_correct;
   int *local_num_correct;
   int *local_cycle_num_correct;
   int *last_read_correct;
   int *last_read_cycle_correct;
   double *mean_grid_wait;
   double *max_grid_wait;
   double *min_grid_wait;
   vector<int> grid_wait_hist;
   int tot_work;
   int *level_work;
   double *frac_level_work;
}GridData;

typedef struct{
   ThreadData thread;
   MatrixData data;
   VectorData vector;
   VectorData *level_vector;
   MatrixData matrix;
   InputData input;
   OutputData output;
   GridData grid;
   PardisoData pardiso;
   MfemData mfem;
}AllData;

#endif
