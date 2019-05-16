#ifndef DMEM_MISC_HPP
#define DMEM_MISC_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_PrintOutput(DMEM_AllData *dmem_all_data);

void DMEM_PrintParCSRMatrix(hypre_ParCSRMatrix *A, char *filename);

HYPRE_Real InnerProd(hypre_Vector *x,
                     hypre_Vector *y,
                     MPI_Comm comm);

#endif
