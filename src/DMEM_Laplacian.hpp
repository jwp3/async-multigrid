#ifndef DMEM_LAPLACIAN_HPP
#define DMEM_LAPLACIAN_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_Laplacian_3D_27pt(DMEM_AllData *dmem_all_data,
                            HYPRE_ParCSRMatrix *A_ptr,
                            MPI_Comm comm,
                            HYPRE_Int nx,
                            HYPRE_Int ny,
                            HYPRE_Int nz);

#endif
