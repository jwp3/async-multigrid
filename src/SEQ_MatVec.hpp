#ifndef SEQ_MATVEC_HPP
#define SEQ_MATVEC_HPP

#include "Main.hpp"

void SEQ_MatVec(AllData *all_data,
                hypre_CSRMatrix *A,
                HYPRE_Real *x_data,
                HYPRE_Real *y_data);

void SEQ_MatVecT(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *x,
                 HYPRE_Real *y);

void SEQ_Residual(AllData *all_data,
                  hypre_CSRMatrix *A,
                  HYPRE_Real *b,
                  HYPRE_Real *x,
                  HYPRE_Real *y,
                  HYPRE_Real *r);

#endif
