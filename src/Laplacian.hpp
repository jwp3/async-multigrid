#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP


#include "Main.hpp"

void Laplacian_2D_5pt(HYPRE_IJMatrix *A, int n);

void MFEM_Ex1(HYPRE_IJMatrix *Aij, int ref_levels);

#endif
