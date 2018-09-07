#ifndef SMEM_SMOOTH_HPP
#define SMEM_SMOOTH_HPP

#include "Main.hpp"

void SMEM_Sync_Parfor_Jacobi(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *f,
                             HYPRE_Real *u,
                             HYPRE_Real *u_prev,
                             int num_sweeps);

void SMEM_SemiAsync_Parfor_GaussSeidel(AllData *all_data,
                                      hypre_CSRMatrix *A,
                                      HYPRE_Real *f,
                                      HYPRE_Real *u,
                                      int num_sweeps);

void SMEM_Sync_Parfor_HybridJacobiGaussSeidel(AllData *all_data,
                                              hypre_CSRMatrix *A,
                                              HYPRE_Real *f,
                                              HYPRE_Real *u,
                                              HYPRE_Real *u_prev,
                                              int num_sweeps,
                                              int level);

void SMEM_Sync_Jacobi(AllData *all_data,
                      hypre_CSRMatrix *A,
                      HYPRE_Real *f,
                      HYPRE_Real *u,
                      HYPRE_Real *u_prev,
                      int num_sweeps,
                      int thread_level,
                      int ns, int ne);

void SMEM_SemiAsync_GaussSeidel(AllData *all_data,
                                hypre_CSRMatrix *A,
                                HYPRE_Real *f,
                                HYPRE_Real *u,
                                int num_sweeps,
                                int thread_level,
                                int ns, int ne);

void SMEM_Sync_HybridJacobiGaussSeidel(AllData *all_data,
                                       hypre_CSRMatrix *A,
                                       HYPRE_Real *f,
                                       HYPRE_Real *u,
                                       HYPRE_Real *u_prev,
                                       int num_sweeps,
                                       int thread_level,
                                       int ns, int ne);

#endif
