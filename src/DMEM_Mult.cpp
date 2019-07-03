#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Add.hpp"

void MultCycle(DMEM_AllData *dmem_all_data,
               HYPRE_Int cycle);
void SyncAddCycle(DMEM_AllData *dmem_all_data,
                  HYPRE_Solver solver,
                  HYPRE_Int cycle);

void DMEM_Mult(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   
   double begin = MPI_Wtime();
   for (dmem_all_data->iter.cycle = 1; dmem_all_data->iter.cycle <= dmem_all_data->input.num_cycles; dmem_all_data->iter.cycle += 1){
      MultCycle(dmem_all_data, dmem_all_data->iter.cycle);
      double residual_begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[0],
                                         U_array[0],
                                         1.0,
                                         F_array[0],
                                         dmem_all_data->vector_fine.r);
      dmem_all_data->output.residual_wtime += MPI_Wtime() - residual_begin;
      double residual_norm_begin = MPI_Wtime();
      hypre_ParVector *r = dmem_all_data->vector_fine.r;
      HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r));
      dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
     // if (my_id == 0) printf("%e\n", res_norm/dmem_all_data->output.r0_norm2);
      if (res_norm/dmem_all_data->output.r0_norm2 < dmem_all_data->input.tol) break;
   }
   dmem_all_data->output.solve_wtime = MPI_Wtime() - begin;
}

void MultCycle(DMEM_AllData *dmem_all_data,
	       HYPRE_Int cycle)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   hypre_ParAMGData *amg_data = 
      (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   double begin;

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

   int coarsest_level = dmem_all_data->grid.num_levels-1;
   if (dmem_all_data->input.solver == MULT_MULTADD){
      coarsest_level = dmem_all_data->input.coarsest_mult_level;
   }

   for (HYPRE_Int level = 0; level < coarsest_level; level++){
      double level_begin = MPI_Wtime();
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;
  
      A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_grid])); 
      hypre_ParVectorCopy(F_array[fine_grid], Vtemp); 
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
      HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      /* smooth */
      begin = MPI_Wtime();
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
      if (dmem_all_data->input.smoother == L1_JACOBI){
         for (HYPRE_Int i = 0; i < num_rows; i++){
            u_local_data[i] += v_local_data[i] / dmem_all_data->matrix.L1_row_norm_fine[fine_grid][i];
         }
      }
      else {
         for (HYPRE_Int i = 0; i < num_rows; i++){
            u_local_data[i] += dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
         }
      }
      dmem_all_data->output.smooth_wtime += MPI_Wtime() - begin;
 
      hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
      /* restrict */
      begin = MPI_Wtime();
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
      dmem_all_data->output.restrict_wtime += MPI_Wtime() - begin;
      dmem_all_data->output.level_wtime[level] += MPI_Wtime() - level_begin;
   }

   double coarse_r0_norm2, res_norm;
   
   begin = MPI_Wtime();
   if (dmem_all_data->input.solver == MULT_MULTADD){
      hypre_ParVectorSetConstantValues(dmem_all_data->vector_fine.x, 0.0);
      hypre_ParVectorSetConstantValues(dmem_all_data->vector_gridk.x, 0.0);
      DMEM_Add(dmem_all_data);
      int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[coarsest_level]));
      HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_fine.x));
      for (HYPRE_Int i = 0; i < num_rows; i++){
         u_local_data[i] = x_local_data[i];
      }
   }
   else {
      hypre_GaussElimSolve(amg_data, coarsest_level, 99);
   }
   dmem_all_data->output.coarsest_solve_wtime += MPI_Wtime() - begin;
    
   for (HYPRE_Int level = coarsest_level; level > 0; level--){
      double level_begin = MPI_Wtime();
      HYPRE_Int fine_grid = level - 1;
      HYPRE_Int coarse_grid = level;

      /* prolong and correct */
      begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvec(1.0,
                               P_array[fine_grid], 
                               U_array[coarse_grid],
                               1.0,
                               U_array[fine_grid]);            
      dmem_all_data->output.prolong_wtime += MPI_Wtime() - begin;
      
      /* smooth */
      begin = MPI_Wtime();
      A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
      hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
      HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      /* smooth */
      hypre_ParCSRMatrixMatvec(-1.0,
                               A_array[fine_grid],
                               U_array[fine_grid],
                               1.0,
                               Vtemp);
      if (dmem_all_data->input.smoother == L1_JACOBI){
         for (HYPRE_Int i = 0; i < num_rows; i++){
            u_local_data[i] += v_local_data[i] / dmem_all_data->matrix.L1_row_norm_fine[fine_grid][i];
         }  
      }
      else {
         for (HYPRE_Int i = 0; i < num_rows; i++){
            u_local_data[i] += dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
         }  
      }
      dmem_all_data->output.smooth_wtime += MPI_Wtime() - begin;
      dmem_all_data->output.level_wtime[level] += MPI_Wtime() - level_begin;
   }
}

void DMEM_SyncAdd(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
      SyncAddCycle(dmem_all_data, dmem_all_data->hypre.solver, cycle);
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[0],
                                         dmem_all_data->vector_fine.x,
                                         1.0,
                                         dmem_all_data->vector_fine.b,
                                         dmem_all_data->vector_fine.r);
      hypre_ParVector *r = dmem_all_data->vector_fine.r;
      HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r));
      if (my_id == 0) printf("%e\n", res_norm/dmem_all_data->output.r0_norm2);
   }
}

void SyncAddCycle(DMEM_AllData *dmem_all_data,
                  HYPRE_Solver solver,
                  HYPRE_Int cycle)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)solver;

   hypre_ParVector *x;

   HYPRE_Int num_rows;

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

   hypre_ParVector *e = dmem_all_data->vector_fine.e;
   HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   hypre_ParVectorSetConstantValues(e, 0.0);

   x = dmem_all_data->vector_fine.x;
   hypre_ParVectorCopy(dmem_all_data->vector_fine.r,
                       F_array[0]);

   HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   for (HYPRE_Int level = 0; level < num_levels-1; level++){
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;

      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
                                F_array[fine_grid],
                                0.0,
                                F_array[coarse_grid]);
   }

   for (HYPRE_Int level = 0; level < num_levels; level++){
      A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[level]));
      A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[level]));
      num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[level]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));

      for (HYPRE_Int i = 0; i < num_rows; i++){
         u_local_data[i] = dmem_all_data->input.smooth_weight * f_local_data[i] / A_data[A_i[i]];
      }

      hypre_ParCSRMatrixMatvec(1.0,
                               A_array[level],
                               U_array[level],
                               0.0,
                               Vtemp);

      for (HYPRE_Int i = 0; i < num_rows; i++){
         u_local_data[i] = 2.0 * u_local_data[i] -
            dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
      }

      for (HYPRE_Int interp_level = level; interp_level > 0; interp_level--){
         HYPRE_Int fine_grid = interp_level - 1;
         HYPRE_Int coarse_grid = interp_level;
         hypre_ParCSRMatrixMatvec(1.0,
                                  P_array[fine_grid],
                                  U_array[coarse_grid],
                                  0.0,
                                  U_array[fine_grid]);
      }

      num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));
      for (HYPRE_Int i = 0; i < num_rows; i++){
         x_local_data[i] += u_local_data[i];
         e_local_data[i] += u_local_data[i];
      }
   }
}
