#ifndef SMEM_MATVEC_HPP
#define SMEM_MATVEC_HPP

#include "Main.hpp"

void SMEM_Sync_Parfor_MatVec(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *x_data,
                             HYPRE_Real *y_data);

void SMEM_Sync_Parfor_MatVecT(AllData *all_data,
                              hypre_CSRMatrix *A,
                              HYPRE_Real *x,
                              HYPRE_Real *y,
                              HYPRE_Real *y_expand);

void SMEM_Sync_Parfor_SpGEMV(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *x,
                             HYPRE_Real *b,
                             HYPRE_Real alpha,
                             HYPRE_Real beta,
                             HYPRE_Real *y);

void SMEM_Sync_Parfor_Residual(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *b,
                               HYPRE_Real *x,
                               HYPRE_Real *y,
                               HYPRE_Real *r);

void SMEM_Sync_SpGEMV(AllData *all_data,
                      hypre_CSRMatrix *A,
                      HYPRE_Real *x,
                      HYPRE_Real *b,
                      HYPRE_Real alpha,
                      HYPRE_Real beta,
                      HYPRE_Real *y);

void SMEM_SpGEMV(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *x,
                 HYPRE_Real *b,
                 HYPRE_Real alpha,
                 HYPRE_Real beta,
                 HYPRE_Real *y,
                 int iBegin, int iEnd);

void SMEM_Sync_Residual(AllData *all_data,
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

#endif
