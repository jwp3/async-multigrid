#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_SyncAdd.hpp"

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
      DMEM_SyncAddCycle(dmem_all_data, dmem_all_data->hypre.solver, cycle);
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
  // MPI_Barrier(MPI_COMM_WORLD);
  // HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
  // HYPRE_Real *r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_fine.r));
  // HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_fine.x));
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       for (HYPRE_Int i = 0; i < num_rows; i++){
  //         // printf("0 %e\n", x_local_data[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void DMEM_SyncAddCycle(DMEM_AllData *dmem_all_data,
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

   if (dmem_all_data->input.solver == ASYNC_MULTADD){
      x = dmem_all_data->vector_gridk.x;
      hypre_ParVectorCopy(dmem_all_data->vector_gridk.r,
                          F_array[0]);
   }
   else {   
      x = dmem_all_data->vector_fine.x; 
      hypre_ParVectorCopy(dmem_all_data->vector_fine.r,
                          F_array[0]);
   }

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
      
     // if (level == 1)
     // for (int p = 0; p < num_procs; p++){
     //    if (my_id == p){
     //       for (HYPRE_Int i = 0; i < num_rows; i++){
     //          printf("%e\n", u_local_data[i]);
     //       }
     //    }
     //    MPI_Barrier(MPI_COMM_WORLD);
     // }

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
     
     // if (cycle == dmem_all_data->input.num_cycles) 
     // for (int p = 0; p < num_procs; p++){
     //    if (my_id == p){
     //       for (HYPRE_Int i = 0; i < num_rows; i++){
     //          printf("%d %e\n", level, u_local_data[i]);
     //       }
     //    }
     //    MPI_Barrier(MPI_COMM_WORLD);
     // }
   }

  // MPI_Barrier(MPI_COMM_WORLD);
  // if (my_id == 0) printf("\n");
  // MPI_Barrier(MPI_COMM_WORLD);
  // if (cycle == dmem_all_data->input.num_cycles)
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       for (HYPRE_Int i = 0; i < num_rows; i++){
  //          printf("0 %e\n", e_local_data[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}
