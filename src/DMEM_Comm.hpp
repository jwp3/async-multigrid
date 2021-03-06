#ifndef DMEM_COMM_HPP
#define DMEM_COMM_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

int SendRecv(DMEM_AllData *dmem_all_data,
             DMEM_CommData *comm_data,
             HYPRE_Real *v,
             HYPRE_Int op);

void CompleteRecv(DMEM_AllData *dmem_all_data,
                  DMEM_CommData *comm_data,
                  HYPRE_Real *v,
                  HYPRE_Int op);

void CompleteInFlight(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);

void CheckInFlight(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int i);

#endif
