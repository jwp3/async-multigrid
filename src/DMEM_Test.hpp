#ifndef DMEM_TEST_HPP
#define DMEM_TEST_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_TestCorrect_LocalRes(DMEM_AllData *dmem_all_data);

void DMEM_TestCorrect_GlobalRes(DMEM_AllData *dmem_all_data);
void DMEM_TestResidual_GlobalRes(DMEM_AllData *dmem_all_data);
void DMEM_TestIsendrecv(DMEM_AllData *dmem_all_data);

#endif
