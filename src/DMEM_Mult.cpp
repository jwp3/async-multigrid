#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"

void MultCycle(DMEM_AllData *dmem_all_data,
               HYPRE_Int cycle);

void DMEM_Mult(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   HYPRE_Int num_cycles = dmem_all_data->input.num_cycles;
   HYPRE_Int start_cycle = dmem_all_data->input.start_cycle;
   HYPRE_Int increment_cycle = dmem_all_data->input.increment_cycle;

   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

   for (HYPRE_Int cycle = start_cycle; cycle <= num_cycles; cycle += increment_cycle){
      MultCycle(dmem_all_data, cycle);
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[0],
                                         U_array[0],
                                         1.0,
                                         F_array[0],
                                         dmem_all_data->vector_fine.r);
      HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(dmem_all_data->vector_fine.r, dmem_all_data->vector_fine.r));
      if (my_id == 0) printf("%e\n", res_norm);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

void MultCycle(DMEM_AllData *dmem_all_data,
	       HYPRE_Int cycle)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   hypre_ParAMGData *amg_data = 
      (hypre_ParAMGData *)dmem_all_data->hypre.solver;

   HYPRE_Real *u_local_data;
   HYPRE_Real *v_local_data;
   HYPRE_Real *f_local_data;
   HYPRE_Real *A_data;
   HYPRE_Int *A_i;

   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);

   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   for (HYPRE_Int level = 0; level < num_levels-1; level++){
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;
  
      A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_grid])); 
      hypre_ParVectorCopy(F_array[fine_grid], Vtemp); 
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
      HYPRE_Int num_rows =
         hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      /* smooth */
      if (level == 0 && cycle > 1){
         hypre_ParCSRMatrixMatvec(-1.0,
                                  A_array[fine_grid],
                                  U_array[fine_grid],
                                  1.0,
                                  Vtemp);
      }
      else {
         hypre_ParVectorSetConstantValues(U_array[fine_grid], 0.0);
      }
      for (HYPRE_Int i = 0; i < num_rows; i++){
         u_local_data[i] +=
            dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
      }
 
      hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
      /* restrict */
      hypre_ParCSRMatrixMatvec(-1.0,
                               A_array[fine_grid],
                               U_array[fine_grid],
                               1.0,
                               Vtemp);
      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
        			Vtemp,
                                0.0,
                                F_array[coarse_grid]);
   }

   HYPRE_Int coarsest_grid = num_levels-1;
   A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[coarsest_grid]));
   A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[coarsest_grid]));
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[coarsest_grid]));
   HYPRE_Int num_rows =
      hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[coarsest_grid]));
   hypre_ParVectorSetConstantValues(U_array[coarsest_grid], 0.0); 
   HYPRE_Int num_relax = 100;
   for (HYPRE_Int k = 0; k < num_relax; k++){
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[coarsest_grid],
                                         U_array[coarsest_grid],
                                         1.0,
				         F_array[coarsest_grid],
                                         Vtemp);
      for (HYPRE_Int i = 0; i < num_rows; i++){
         u_local_data[i] +=
            dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
      }
   }
  // hypre_GaussElimSolve(amg_data, 0, 99);
   
   for (HYPRE_Int level = num_levels-1; level > 0; level--){
      HYPRE_Int fine_grid = level - 1;
      HYPRE_Int coarse_grid = level;

      /* prolong and correct */
      hypre_ParCSRMatrixMatvec(1.0,
                               P_array[fine_grid], 
                               U_array[coarse_grid],
                               1.0,
                               U_array[fine_grid]);            
      
      /* smooth */
      A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
      hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
      HYPRE_Int num_rows =
         hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      /* smooth */
      hypre_ParCSRMatrixMatvec(-1.0,
                               A_array[fine_grid],
                               U_array[fine_grid],
                               1.0,
                               Vtemp);
      for (HYPRE_Int i = 0; i < num_rows; i++){
         u_local_data[i] +=
            dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
      }
   }
}
