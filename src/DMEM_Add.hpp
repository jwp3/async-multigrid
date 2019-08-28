#ifndef DMEM_ADD_HPP
#define DMEM_ADD_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_Add(DMEM_AllData *dmem_all_data);

void DMEM_AddCorrect_LocalRes(DMEM_AllData *dmem_all_data);
void DMEM_AddResidual_LocalRes(DMEM_AllData *dmem_all_data);
void DMEM_SolutionToFinest_LocalRes(DMEM_AllData *dmem_all_data,
                                    hypre_ParVector *x_gridk,
                                    hypre_ParVector *x_fine);
void DMEM_VectorToGridk_LocalRes(DMEM_AllData *dmem_all_data,
                                 hypre_ParVector *r,
                                 hypre_ParVector *f);

void DMEM_AddCorrect_GlobalRes(DMEM_AllData *dmem_all_data);
void DMEM_AddResidual_GlobalRes(DMEM_AllData *dmem_all_data);

#endif
