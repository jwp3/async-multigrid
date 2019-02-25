#ifndef DMEM_COMM_HPP
#define DMEM_COMM_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void GridkSendRecv(DMEM_AllData *dmem_all_data,
                   DMEM_CommData *comm_data,
                   HYPRE_Real *v);

void FineSendRecv(DMEM_AllData *dmem_all_data,
                  DMEM_CommData *comm_data,
                  HYPRE_Real *v);

void CompleteRecv(DMEM_AllData *dmem_all_data,
                  DMEM_CommData *comm_data,
                  HYPRE_Real *v);

#endif
