#ifndef ELASTICITY_HPP
#define ELASTICITY_HPP

#include "Main.hpp"

void Laplacian_2D_5pt(HYPRE_IJMatrix *A,
                      int n);

void Laplacian_3D_27pt(HYPRE_ParCSRMatrix *A_ptr,
                       int n);

void MFEM_Laplacian(AllData *all_data,
                    HYPRE_IJMatrix *Aij);

void MFEM_Elasticity(AllData *all_data,
                     HYPRE_IJMatrix *Aij);

#endif
