#ifndef SEQ_SMOOTH_HPP
#define SEQ_SMOOTH_HPP

#include "Main.hpp"

void SEQ_Jacobi(AllData *all_data,
                hypre_CSRMatrix *A,
                HYPRE_Real *f_data,
                HYPRE_Real *u_data,
                HYPRE_Real *u_data_prev,
                int num_sweeps);

void SEQ_GaussSeidel(AllData *all_data,
                     hypre_CSRMatrix *A,
                     HYPRE_Real *f,
                     HYPRE_Real *u,
                     int num_sweeps);

#endif
