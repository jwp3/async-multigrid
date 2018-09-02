#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP


#include "Main.hpp"

void Laplacian_2D_5pt(HYPRE_IJMatrix *A,
                      int n,
		      int N,
                      int ilower,
                      int iupper);

#endif
