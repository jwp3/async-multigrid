#ifndef DMEM_PARMFEM_HPP
#define DMEM_PARMFEM_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_ParMfem(DMEM_AllData *dmem_all_data,
		  HYPRE_ParCSRMatrix *A,
                  MPI_Comm comm);

#endif
