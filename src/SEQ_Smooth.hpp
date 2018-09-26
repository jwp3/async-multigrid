#ifndef SEQ_SMOOTH_HPP
#define SEQ_SMOOTH_HPP

#include "Main.hpp"

void SEQ_Jacobi(AllData *all_data,
                hypre_CSRMatrix *A,
                HYPRE_Real *f_data,
                HYPRE_Real *u_data,
                HYPRE_Real *u_data_prev,
                int num_sweeps);

void SEQ_GaussSeidel(AllData *all_data,
                     hypre_CSRMatrix *A,
                     HYPRE_Real *f,
                     HYPRE_Real *u,
                     int num_sweeps);

void SEQ_L1Jacobi(AllData *all_data,
                  hypre_CSRMatrix *A,
                  HYPRE_Real *f,
                  HYPRE_Real *u,
                  HYPRE_Real *u_prev,
		  double *L1_row_norm,
                  int num_sweeps);

void SEQ_SymmetricJacobi(AllData *all_data,
                         hypre_CSRMatrix *A,
                         HYPRE_Real *f,
                         HYPRE_Real *u,
                         HYPRE_Real *y,
                         HYPRE_Real *r,
                         int num_sweeps,
                         int level);

void SEQ_SymmetricL1Jacobi(AllData *all_data,
                           hypre_CSRMatrix *A,
                           HYPRE_Real *f,
                           HYPRE_Real *u,
                           HYPRE_Real *y,
                           HYPRE_Real *r,
                           int num_sweeps,
                           int level);

#endif
