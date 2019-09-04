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
void DMEM_AddCheckComm(DMEM_AllData *dmem_all_data);

void DMEM_AddCorrect_GlobalRes(DMEM_AllData *dmem_all_data);
void DMEM_AddResidual_GlobalRes(DMEM_AllData *dmem_all_data);

int DMEM_CheckOutsideDoneFlag(DMEM_AllData *dmem_all_data);
void DMEM_AsyncRecvStart(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);

int DMEM_CheckMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);
int DMEM_CheckMessageFlagsValue(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int value);
int DMEM_CheckMessageFlagsNotValue(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int value);

#endif
