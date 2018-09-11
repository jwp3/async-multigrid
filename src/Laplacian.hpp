#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP


#include "Main.hpp"

void Laplacian_2D_5pt(HYPRE_IJMatrix *A,
                      int n);

void MFEM_Ex1(AllData *all_data,
              HYPRE_IJMatrix *Aij);

#endif
