#ifndef DMEM_SMOOTH_HPP
#define DMEM_SMOOTH_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_AddSmooth(DMEM_AllData *dmem_all_data, int coarsest_level);

void DMEM_AsyncSmooth(DMEM_AllData *dmem_all_data, int level);

void DMEM_AsyncSmoothEnd(DMEM_AllData *dmem_all_data);

#endif
