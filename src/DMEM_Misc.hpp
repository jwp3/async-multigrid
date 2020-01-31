#ifndef DMEM_MISC_HPP
#define DMEM_MISC_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_PrintOutput(DMEM_AllData *dmem_all_data);

void DMEM_PrintParCSRMatrix(hypre_ParCSRMatrix *A, char *filename);

void DMEM_ResetAllCommData(DMEM_AllData *dmem_all_data);

HYPRE_Real InnerProd(hypre_Vector *x,
                     hypre_Vector *y,
                     MPI_Comm comm);

HYPRE_Real InnerProdFlag(hypre_Vector *x_local,
                         hypre_Vector *y_local,
                         MPI_Comm comm,
                         HYPRE_Real my_flag,
                         HYPRE_Real *sum_flags);

void DMEM_HypreParVector_Ivaxpy(hypre_ParVector *y, hypre_ParVector *x, HYPRE_Complex *scale, HYPRE_Int size);

void DMEM_HypreParVector_Axpy(hypre_ParVector *y, hypre_ParVector *x, HYPRE_Complex alpha, HYPRE_Int size);

void DMEM_HypreParVector_Copy(hypre_ParVector *y, hypre_ParVector *x, HYPRE_Int size);

void DMEM_HypreParVector_Set(hypre_ParVector *y, HYPRE_Complex alpha, HYPRE_Int size);

void DMEM_HypreParVector_Scale(hypre_ParVector *y, HYPRE_Complex alpha, HYPRE_Int size);

void DMEM_HypreRealArray_Copy(HYPRE_Real *y, HYPRE_Real *x, HYPRE_Int size);

void DMEM_HypreRealArray_Axpy(HYPRE_Real *y, HYPRE_Real *x, HYPRE_Real alpha, HYPRE_Int size);

void DMEM_HypreRealArray_Set(HYPRE_Real *y, HYPRE_Real alpha, HYPRE_Int size);

void DMEM_HypreRealArray_Prefetch(HYPRE_Real *y, HYPRE_Int size, HYPRE_Int to_location);

void DMEM_WriteCSR(CSR A, char *out_str, int base, OrderingData P, MPI_Comm comm);

#endif
