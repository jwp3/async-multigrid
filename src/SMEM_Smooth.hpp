#ifndef SMEM_SMOOTH_HPP
#define SMEM_SMOOTH_HPP

#include "Main.hpp"

void SMEM_Sync_Parfor_Jacobi(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *f,
                             HYPRE_Real *u,
                             HYPRE_Real *u_prev,
                             int num_sweeps,
                             int level);

void SMEM_Sync_Parfor_L1Jacobi(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *f,
                               HYPRE_Real *u,
                               HYPRE_Real *u_prev,
                               int num_sweeps,
                               int level);

void SMEM_SemiAsync_Parfor_GaussSeidel(AllData *all_data,
                                      hypre_CSRMatrix *A,
                                      HYPRE_Real *f,
                                      HYPRE_Real *u,
                                      int num_sweeps,
				      int level);

void SMEM_Async_Parfor_GaussSeidel(AllData *all_data,
                                   hypre_CSRMatrix *A,
                                   HYPRE_Real *f,
                                   HYPRE_Real *u,
                                   int num_sweeps,
                                   int level);

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
                      int level,
                      int ns, int ne);

void SMEM_Sync_L1Jacobi(AllData *all_data,
                        hypre_CSRMatrix *A,
                        HYPRE_Real *f,
                        HYPRE_Real *u,
                        HYPRE_Real *u_prev,
                        int num_sweeps,
                        int level,
                        int ns, int ne);

void SMEM_SemiAsync_GaussSeidel(AllData *all_data,
                                hypre_CSRMatrix *A,
                                HYPRE_Real *f,
                                HYPRE_Real *u,
                                int num_sweeps,
                                int level,
                                int ns, int ne);

void SMEM_Async_GaussSeidel(AllData *all_data,
                            hypre_CSRMatrix *A,
                            HYPRE_Real *f,
                            HYPRE_Real *u,
                            int num_sweeps,
                            int level,
                            int ns, int ne);

void SMEM_Sync_HybridJacobiGaussSeidel(AllData *all_data,
                                       hypre_CSRMatrix *A,
                                       HYPRE_Real *f,
                                       HYPRE_Real *u,
                                       HYPRE_Real *u_prev,
                                       int num_sweeps,
                                       int level,
                                       int ns, int ne);

void SMEM_Sync_SymmetricJacobi(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *f,
                               HYPRE_Real *u,
                               HYPRE_Real *y,
                               HYPRE_Real *r,
			       int num_sweeps,
                               int level,
                               int ns, int ne);

void SMEM_Sync_SymmetricL1Jacobi(AllData *all_data,
                                 hypre_CSRMatrix *A,
                                 HYPRE_Real *f,
                                 HYPRE_Real *u,
                                 HYPRE_Real *y,
                                 HYPRE_Real *r,
                                 int num_sweeps,
                                 int level,
                                 int ns, int ne);

#endif
