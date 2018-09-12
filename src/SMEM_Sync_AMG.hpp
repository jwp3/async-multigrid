#ifndef SMEM_SYNC_AMG_HPP
#define SMEM_SYNC_AMG_HPP

#include "Main.hpp"

void SMEM_Sync_Parfor_Vcycle(AllData *all_data);

void SMEM_Sync_Parfor_AFACx_Vcycle(AllData *all_data);

//void SMEM_Sync_AFACx_Vcycle(AllData *all_data);

//void SMEM_Sync_Multadd_Vcycle(AllData *all_data);

void SMEM_Sync_Add_Vcycle(AllData *all_data);

#endif
