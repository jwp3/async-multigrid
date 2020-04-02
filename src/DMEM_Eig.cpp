#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Add.hpp"
#include "_hypre_utilities.h"
#include "DMEM_Misc.hpp"
#include "DMEM_Mult.hpp"

void DMEM_PowerMult(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->input.eig_power_max_iters <= 0) return;
   int eig_power_iters;
   double eig_max, eig_min;
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   HYPRE_Real u_norm, y_norm;

  // srand((double)my_id * MPI_Wtime());
   srand((double)my_id);

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);

   hypre_ParCSRMatrix *A = A_array[0];
   hypre_ParVector *u = U_array[0];
   hypre_ParVector *f = F_array[0];
   hypre_ParVector *r = dmem_all_data->vector_fine.r;
   hypre_ParVector *x = dmem_all_data->vector_fine.x;
   hypre_ParVector *y = dmem_all_data->vector_fine.y;
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);

   HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(u));

   HYPRE_BoomerAMGSetMaxIter(dmem_all_data->hypre.solver, dmem_all_data->input.eig_power_MG_max_iters);
   HYPRE_BoomerAMGSetPrintLevel(dmem_all_data->hypre.solver, 0);

   for (int i = 0; i < num_rows; i++) u_local_data[i] = RandDouble(0.0, 1.0)-.5;
   eig_power_iters = 0;
   while (1){
      u_norm = sqrt(hypre_ParVectorInnerProd(u, u));
      DMEM_HypreParVector_Scale(u, 1.0/u_norm, num_rows);
      DMEM_HypreParVector_Copy(y, u, num_rows);
      hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, r);
      DMEM_HypreParVector_Copy(f, r, num_rows);
      DMEM_HypreParVector_Set(u, 0.0, num_rows);
     // if (dmem_all_data->input.solver == MULT){
         HYPRE_BoomerAMGSolve(dmem_all_data->hypre.solver, A, f, u);
     // }
     // else {
     //    DMEM_SyncAddCycle(dmem_all_data);
     // }

     // DMEM_HypreParVector_Copy(f, r, num_rows);
     // DMEM_MultCycle(dmem_all_data);

     // DMEM_HypreParVector_Copy(u, r, num_rows);

      eig_power_iters++;
      if (eig_power_iters == dmem_all_data->input.eig_power_max_iters) break;
   }
   eig_max = hypre_ParVectorInnerProd(y, u);

   for (int i = 0; i < num_rows; i++) u_local_data[i] = RandDouble(0.0, 1.0)-.5;
   eig_power_iters = 0;
   while (1){
      HYPRE_Real u_norm = sqrt(hypre_ParVectorInnerProd(u, u));
      DMEM_HypreParVector_Scale(u, 1.0/u_norm, num_rows);
      DMEM_HypreParVector_Copy(y, u, num_rows);
      hypre_ParCSRMatrixMatvec(1.0, A, u, 0, r);
      DMEM_HypreParVector_Copy(f, r, num_rows);
      DMEM_HypreParVector_Set(u, 0.0, num_rows);
     // if (dmem_all_data->input.solver == MULT){
         HYPRE_BoomerAMGSolve(dmem_all_data->hypre.solver, A, f, u);
     // }
     // else {
     //    DMEM_SyncAddCycle(dmem_all_data);
     // }

     // DMEM_HypreParVector_Copy(f, r, num_rows);
     // DMEM_MultCycle(dmem_all_data);

     // DMEM_HypreParVector_Copy(u, r, num_rows);

      eig_power_iters++;
      if (eig_power_iters == dmem_all_data->input.eig_power_max_iters) break;

      DMEM_HypreParVector_Axpy(u, y, -eig_max, num_rows);
   } 
   eig_min = hypre_ParVectorInnerProd(y, u);

   dmem_all_data->cheby.beta = eig_max - dmem_all_data->input.b_eig_shift;
   dmem_all_data->cheby.alpha = eig_min + dmem_all_data->input.a_eig_shift;

   HYPRE_BoomerAMGSetMaxIter(dmem_all_data->hypre.solver, dmem_all_data->input.num_cycles);
   HYPRE_BoomerAMGSetPrintLevel(dmem_all_data->hypre.solver, dmem_all_data->hypre.print_level);
   srand(0);
   if (my_id == 0 && dmem_all_data->input.oneline_output_flag == 0)
      printf("CHEBY: power eig max %.16e, eig min %.16e\n",
             dmem_all_data->cheby.beta,  dmem_all_data->cheby.alpha);
}
