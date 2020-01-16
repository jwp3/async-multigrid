#ifndef DMEM_BUILD_MATRIX_HPP
#define DMEM_BUILD_MATRIX_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_BuildHypreMatrix(DMEM_AllData *dmem_all_data,
                           HYPRE_ParCSRMatrix *A_ptr,
                           HYPRE_ParVector *rhs_ptr,
                           MPI_Comm comm,
                           HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                           HYPRE_Real cx, HYPRE_Real cy, HYPRE_Real cz,
                           HYPRE_Real ax, HYPRE_Real ay, HYPRE_Real az,
                           HYPRE_Real eps,
                           int atype);

void DMEM_BuildMfemMatrix(DMEM_AllData *dmem_all_data,
                          hypre_ParCSRMatrix **A_ptr,
                          hypre_ParVector **b_ptr,
                          MPI_Comm comm);

void DMEM_MatrixFromFile(char *mat_file_str, hypre_ParCSRMatrix **A_ptr, MPI_Comm comm);

void DMEM_DistributeHypreParCSRMatrix_FineToGridk(DMEM_AllData *dmem_all_data,
                                                  hypre_ParCSRMatrix *A,
                                                  hypre_ParCSRMatrix **B);

void DMEM_DistributeCSR_RootToFine(CSR A,
                                   CSR *B,
                                   OrderingData *P,
                                   MPI_Comm comm);

#endif
