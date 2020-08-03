#ifndef DMEM_MULT_HPP
#define DMEM_MULT_HPP

#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_Mult(DMEM_AllData *dmem_all_data);

void DMEM_SyncAdd(DMEM_AllData *dmem_all_data);

void DMEM_MultCycle(void *amg_vdata,
                    hypre_ParCSRMatrix *A,
                    hypre_ParVector *f,
                    hypre_ParVector *u);

void DMEM_SyncAddCycle(void *amg_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector *f,
                       hypre_ParVector *u);

void DMEM_SyncAFACCycle(void *amg_vdata,
                        hypre_ParCSRMatrix *A,
                        hypre_ParVector *f,
                        hypre_ParVector *u);

extern int cycle_type;
extern int precond_zero_init_guess;

#endif
