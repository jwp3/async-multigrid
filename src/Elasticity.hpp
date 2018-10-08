#ifndef ELASTICITY_HPP
#define ELASTICITY_HPP

#include "Main.hpp"

void MFEM_Elasticity(AllData *all_data,
                     HYPRE_IJMatrix *Aij);

void MFEM_Elasticity2(AllData *all_data,
                      HYPRE_IJMatrix *Aij);

#endif
