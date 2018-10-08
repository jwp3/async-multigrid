#ifndef SMEM_MATVEC_HPP
#define SMEM_MATVEC_HPP

#include "Main.hpp"

void SMEM_Sync_Parfor_MatVec(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *x_data,
                             HYPRE_Real *y_data);

void SMEM_Sync_Parfor_Residual(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *b,
                               HYPRE_Real *x,
                               HYPRE_Real *y,
                               HYPRE_Real *r);

void SMEM_Async_Parfor_MatVec(AllData *all_data,
                              hypre_CSRMatrix *A,
                              HYPRE_Real *x_data,
                              HYPRE_Real *y_data);

void SMEM_Async_Parfor_Residual(AllData *all_data,
                                hypre_CSRMatrix *A,
                                HYPRE_Real *b,
                                HYPRE_Real *x,
                                HYPRE_Real *y,
                                HYPRE_Real *r);

void SMEM_MatVec(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *x,
                 HYPRE_Real *y,
                 int ns, int ne);

void SMEM_Residual(AllData *all_data,
                   hypre_CSRMatrix *A,
                   HYPRE_Real *b,
                   HYPRE_Real *x,
                   HYPRE_Real *y,
                   HYPRE_Real *r,
                   int ns, int ne);

void SMEM_JacobiIterMat_MatVec(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *y,
                               HYPRE_Real *r,
                               int ns, int ne,
                               int thread_level);

//void SMEM_JacobiSymmIterMat_MatVec(AllData *all_data,
//                                   hypre_CSRMatrix *A,
//                                   HYPRE_Real *y,
//                                   HYPRE_Real *r,
//                                   int ns, int ne,
//                                   int thread_level);

#endif
