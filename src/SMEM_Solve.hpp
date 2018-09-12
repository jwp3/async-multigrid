#ifndef SMEM_SOLVE_HPP
#define SMEM_SOLVE_HPP

#include "Main.hpp"

void SMEM_Solve(AllData *all_data);

void SMEM_Smooth(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *f,
                 HYPRE_Real *u,
                 HYPRE_Real *y,
                 HYPRE_Real *r,
                 int num_sweeps,
                 int level,
                 int ns, int ne);

#endif
