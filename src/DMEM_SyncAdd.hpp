#ifndef DMEM_SYNCADD_HPP
#define DMEM_SYNCADD_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_SyncAdd(DMEM_AllData *dmem_all_data);

void DMEM_SyncAddCycle(DMEM_AllData *dmem_all_data,
                       HYPRE_Solver solver,
                       HYPRE_Int cycle);

#endif
