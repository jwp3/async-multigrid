#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Test.hpp"
#include "DMEM_Add.hpp"
#include "DMEM_Smooth.hpp"

void AddCycle(DMEM_AllData *dmem_all_data);
void FineSmooth(DMEM_AllData *dmem_all_data);
void PrintMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);
void AsyncStart(DMEM_AllData *dmem_all_data);
void AsyncEnd(DMEM_AllData *dmem_all_data);
int CheckConverge(DMEM_AllData *dmem_all_data);

void DMEM_Add(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // DMEM_TestCorrect_LocalRes(dmem_all_data);
  // return;

   hypre_ParVector *r;

   dmem_all_data->iter.r_norm2_local_converge_flag = 0;
   dmem_all_data->comm.all_done_flag = 0;
   dmem_all_data->comm.outside_done_flag = 0;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   DMEM_ResetAllCommData(dmem_all_data);

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParAMGData *amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   MPI_Comm comm_gridk = dmem_all_data->grid.my_comm;
   hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);
   hypre_ParVector **F_array_fine = hypre_ParAMGDataFArray(amg_data_fine);
   hypre_ParCSRMatrix **A_array_gridk = hypre_ParAMGDataAArray(amg_data_gridk);
   hypre_ParCSRMatrix **A_array_fine = hypre_ParAMGDataAArray(amg_data_fine);
   HYPRE_Int num_rows_gridk = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_gridk[finest_level]));
   HYPRE_Int num_rows_fine = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_fine[finest_level]));
  
   double begin = MPI_Wtime();

   if (dmem_all_data->input.res_compute_type == LOCAL_RES){
     // DMEM_VectorToGridk_LocalRes(dmem_all_data, F_array_fine[finest_level], F_array_gridk[finest_level]);
      DMEM_VectorToGridk_LocalRes(dmem_all_data, dmem_all_data->vector_fine.r, F_array_gridk[finest_level]);
      DMEM_VectorToGridk_LocalRes(dmem_all_data, dmem_all_data->vector_fine.x, dmem_all_data->vector_gridk.x);
      DMEM_VectorToGridk_LocalRes(dmem_all_data, dmem_all_data->vector_fine.b, dmem_all_data->vector_gridk.b);
   }
   if (dmem_all_data->input.async_flag == 1){
      AsyncStart(dmem_all_data);
   }

   double tol;
   HYPRE_Real r0_norm2;
   if (dmem_all_data->input.solver == MULT_MULTADD){
      dmem_all_data->iter.inner_cycle = 0;
      dmem_all_data->input.check_res_flag = 0;
      tol = dmem_all_data->input.inner_tol;
      r0_norm2 = sqrt(hypre_ParVectorInnerProd(F_array_gridk[finest_level], F_array_gridk[finest_level]));
   }
   else {
      dmem_all_data->iter.cycle = 0;
      r0_norm2 = dmem_all_data->output.r0_norm2;
      tol = dmem_all_data->input.tol;
   }

   int enter_add_loop_flag = 1;
   dmem_all_data->output.start_wtime += MPI_Wtime() - begin;
   if (my_grid == 0){
      if (dmem_all_data->input.smoother == ASYNC_JACOBI){
         hypre_ParVectorCopy(F_array_gridk[0], dmem_all_data->vector_gridk.r);
         DMEM_AsyncSmooth(dmem_all_data, 0);
        // if (dmem_all_data->input.async_flag == 0){
            enter_add_loop_flag = 0;
        // }      
      }
   }
   
   while (enter_add_loop_flag){
      int converge_flag = CheckConverge(dmem_all_data);
      AddCycle(dmem_all_data);

      if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
         DMEM_AddCorrect_GlobalRes(dmem_all_data);
         FineSmooth(dmem_all_data);
         double residual_begin = MPI_Wtime();
         DMEM_AddCorrect_GlobalRes(dmem_all_data);
         dmem_all_data->output.residual_wtime += MPI_Wtime() - residual_begin;         
      }
      else {
         DMEM_AddCorrect_LocalRes(dmem_all_data);
         double residual_begin = MPI_Wtime();
         DMEM_AddResidual_LocalRes(dmem_all_data);
         dmem_all_data->output.residual_wtime += MPI_Wtime() - residual_begin;
      }
 
      if (dmem_all_data->input.check_res_flag == 1){
        // if (((dmem_all_data->input.res_compute_type == LOCAL_RES) || (dmem_all_data->input.res_compute_type == GLOBAL_RES && dmem_all_data->input.async_flag == 1)) && 
        //     (dmem_all_data->iter.r_norm2_local_converge_flag == 0)){
         if (dmem_all_data->input.res_compute_type == LOCAL_RES){
            double residual_norm_begin = MPI_Wtime();
            hypre_Vector *f = hypre_ParVectorLocalVector(F_array_gridk[finest_level]);
            dmem_all_data->iter.r_norm2_local = sqrt(InnerProd(f, f, comm_gridk))/r0_norm2;
            dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
            if (dmem_all_data->iter.r_norm2_local < tol){
               dmem_all_data->iter.r_norm2_local_converge_flag = 1;
            }
         }
        // else if (dmem_all_data->input.res_compute_type == GLOBAL_RES && dmem_all_data->input.async_flag == 0){
         else {
            double residual_norm_begin = MPI_Wtime();
            r = dmem_all_data->vector_fine.r;
            HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r))/r0_norm2;
            dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
            if (res_norm < tol){
               dmem_all_data->iter.r_norm2_local_converge_flag = 1;
            }
         }
      }
     
      if (dmem_all_data->input.solver == MULT_MULTADD){
         dmem_all_data->iter.inner_cycle += 1;
      }
      else {
         dmem_all_data->iter.cycle += 1;
      }

      if (converge_flag == 1){
          break;
      }   
   }

  // if (dmem_all_data->input.solver == MULT_MULTADD){
  //    dmem_all_data->output.inner_solve_wtime += MPI_Wtime() - begin;
  // }
  // else {
  //    dmem_all_data->output.solve_wtime = MPI_Wtime() - begin;
  // }

   double end_begin = MPI_Wtime();
   if (dmem_all_data->input.async_flag == 1){
      AsyncEnd(dmem_all_data);
   }
   dmem_all_data->output.end_wtime += MPI_Wtime() - end_begin;

   if (dmem_all_data->input.solver == MULT_MULTADD){
      dmem_all_data->output.inner_solve_wtime += MPI_Wtime() - begin;
   }
   else {
      dmem_all_data->output.solve_wtime = MPI_Wtime() - begin;
   }

   if (dmem_all_data->input.res_compute_type == LOCAL_RES){
      DMEM_SolutionToFinest_LocalRes(dmem_all_data, dmem_all_data->vector_gridk.x, dmem_all_data->vector_fine.x);
      if (dmem_all_data->input.solver == MULTADD ||
          dmem_all_data->input.solver == BPX ||
          dmem_all_data->input.solver == AFACX){

         hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                            dmem_all_data->matrix.A_fine,
                                            dmem_all_data->vector_fine.x,
                                            1.0,
                                            dmem_all_data->vector_fine.b,
                                            dmem_all_data->vector_fine.r);
         hypre_ParCSRMatrixMatvec(1.0,
                                  dmem_all_data->matrix.A_fine,
                                  dmem_all_data->vector_fine.x,
                                  0.0,
                                  dmem_all_data->vector_fine.e);
      }
   }
   else {
      if (dmem_all_data->input.async_flag == 1){
     //    hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
     //                                       dmem_all_data->matrix.A_fine,
     //                                       dmem_all_data->vector_fine.x,
     //                                       1.0,
     //                                       dmem_all_data->vector_fine.b,
     //                                       dmem_all_data->vector_fine.r);
     //   // printf("%d\n", my_id);
      }
   }
   if (dmem_all_data->input.solver == MULT_MULTADD){
      dmem_all_data->input.check_res_flag = 1;
   }

  // for (HYPRE_Int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("id %d, grid %d:\n", my_id, my_grid);
  //       HYPRE_Real *x = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x));
  //       for (int i = 0; i < num_rows_gridk; i++){
  //          printf("\t%e\n", x[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void AddCycle(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   HYPRE_Int num_rows;
   int fine_level, coarse_level;
   double begin;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;

   HYPRE_Real *u_local_data;
   HYPRE_Real *v_local_data;
   HYPRE_Real *f_local_data;

   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);

   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   int num_levels = dmem_all_data->grid.num_levels;
   int my_grid = dmem_all_data->grid.my_grid;

//   num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
//   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
//  // if (my_id == 1) usleep(10000);
//   for (HYPRE_Int i = 0; i < num_rows; i++){
//      if (my_id == 0){
//         u_local_data[i] = 2.0;
//      }
//      else {
//         u_local_data[i] = 1.0;
//      }
//   }
//   return;

   int finest_level = dmem_all_data->input.coarsest_mult_level;
   int coarsest_level;
   if (my_grid < num_levels-1 && dmem_all_data->input.solver == AFACX){
      coarsest_level = my_grid + 1;
   }
   else {
      coarsest_level = my_grid;
   }

   if (dmem_all_data->input.num_interpolants == ONE_INTERPOLANT){
      if (my_grid > 0){
         begin = MPI_Wtime();
         hypre_ParCSRMatrixMatvecT(1.0,
                                   dmem_all_data->matrix.R_gridk,
                                   F_array[0],
                                   0.0,
                                   F_array[coarsest_level]);
         dmem_all_data->output.prolong_wtime += MPI_Wtime() - begin;
         if (dmem_all_data->input.async_flag == 1){
            DMEM_AddCheckComm(dmem_all_data);
         }
      }
   }
   else {
      for (int level = finest_level; level < coarsest_level; level++){
         double level_begin = MPI_Wtime();
         fine_level = level;
         coarse_level = level + 1;
         begin = MPI_Wtime();
         hypre_ParCSRMatrixMatvecT(1.0,
                                   R_array[fine_level],
                                   F_array[fine_level],
                                   0.0,
                                   F_array[coarse_level]);
         dmem_all_data->output.restrict_wtime += MPI_Wtime() - begin;
         if (dmem_all_data->input.async_flag == 1){
            DMEM_AddCheckComm(dmem_all_data);
         }
         dmem_all_data->output.level_wtime[level] += MPI_Wtime() - level_begin;
      }
   }

   HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
   HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
   num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
   f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[coarsest_level]));
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[coarsest_level]));

   begin = MPI_Wtime();
   if (my_grid == num_levels-1){
      begin = MPI_Wtime();
      hypre_GaussElimSolve(amg_data, coarsest_level, 99);
      dmem_all_data->output.coarsest_solve_wtime += MPI_Wtime() - begin;
   }
   else {
     // begin = MPI_Wtime();
      if (dmem_all_data->input.solver == BPX){
         double prob = RandDouble(0, 1.0);
         for (HYPRE_Int i = 0; i < num_rows; i++){
            if (RandDouble(0, 1.0) < prob){
               u_local_data[i] = dmem_all_data->input.smooth_weight * f_local_data[i] / A_data[A_i[i]];
            }
         }
      }
      else if (dmem_all_data->input.solver == AFACX){
         hypre_ParVectorCopy(F_array[coarsest_level], Vtemp);
         hypre_ParVectorSetConstantValues(U_array[coarsest_level], 0.0);
         for (int k = 0; k < dmem_all_data->input.num_coarse_smooth_sweeps; k++){
            for (int i = 0; i < num_rows; i++){
               u_local_data[i] += dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
            }
            if (k == dmem_all_data->input.num_coarse_smooth_sweeps-1) break;
            hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                               A_array[coarsest_level],
                                               U_array[coarsest_level],
                                               1.0,
                                               F_array[coarsest_level],
                                               Vtemp);
         }
         coarsest_level -= 1;
         hypre_ParCSRMatrixMatvec(1.0,
                                  P_array[coarsest_level],
                                  U_array[coarsest_level+1],
                                  0.0,
                                  U_array[coarsest_level]);
         u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[coarsest_level]));
         A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
         A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
         num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));

         hypre_ParVector *e = dmem_all_data->vector_gridk.e;
         HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
         hypre_ParVectorSetConstantValues(e, 0.0);
         hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                            A_array[coarsest_level],
                                            U_array[coarsest_level],
                                            1.0,
                                            F_array[coarsest_level],
                                            e);
         hypre_ParVectorCopy(e, Vtemp);
         hypre_ParVectorSetConstantValues(U_array[coarsest_level], 0.0);
         for (int k = 0; k < dmem_all_data->input.num_fine_smooth_sweeps; k++){
            for (int i = 0; i < num_rows; i++){
               u_local_data[i] += dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
            }
            if (k == dmem_all_data->input.num_fine_smooth_sweeps-1) break;
            hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                               A_array[coarsest_level],
                                               U_array[coarsest_level],
                                               1.0,
                                               e,
                                               Vtemp);
         }
      }
      else {
         if (dmem_all_data->input.smoother == L1_JACOBI){
            for (HYPRE_Int i = 0; i < num_rows; i++){
               u_local_data[i] = f_local_data[i] / dmem_all_data->matrix.L1_row_norm_gridk[coarsest_level][i];
            }
         }
         else {
            for (HYPRE_Int i = 0; i < num_rows; i++){
               u_local_data[i] = dmem_all_data->input.smooth_weight * f_local_data[i] / A_data[A_i[i]];
            }
         }
        // if (my_grid > 0){
         hypre_ParCSRMatrixMatvec(1.0,
                                  A_array[coarsest_level],
                                  U_array[coarsest_level],
                                  0.0,
                                  Vtemp);
         if (dmem_all_data->input.smoother == L1_JACOBI){
            for (HYPRE_Int i = 0; i < num_rows; i++){
               u_local_data[i] = 2.0 * u_local_data[i] - v_local_data[i] / dmem_all_data->matrix.L1_row_norm_gridk[coarsest_level][i];
            }
         }
         else {
            for (HYPRE_Int i = 0; i < num_rows; i++){
               u_local_data[i] = 2.0 * u_local_data[i] - dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
            }
         }
        // }
      }
     // dmem_all_data->output.smooth_wtime += MPI_Wtime() - begin;
   }
   dmem_all_data->output.smooth_wtime += MPI_Wtime() - begin;
  
   if (dmem_all_data->input.async_flag == 1){
      DMEM_AddCheckComm(dmem_all_data);
   }
   if (dmem_all_data->input.num_interpolants == ONE_INTERPOLANT){
      if (my_grid > 0){
         begin = MPI_Wtime();
         hypre_ParCSRMatrixMatvec(1.0,
                                  dmem_all_data->matrix.P_gridk,
                                  U_array[coarsest_level],
                                  0.0,
                                  U_array[0]);
         dmem_all_data->output.prolong_wtime += MPI_Wtime() - begin;
         if (dmem_all_data->input.async_flag == 1){
            DMEM_AddCheckComm(dmem_all_data);
         }
      }
   }
   else {
      for (int level = coarsest_level; level > finest_level; level--){
         double level_begin = MPI_Wtime();
         fine_level = level-1;
         coarse_level = level;
         begin = MPI_Wtime();
         hypre_ParCSRMatrixMatvec(1.0,
                                  P_array[fine_level], 
                                  U_array[coarse_level],
                                  0.0,
                                  U_array[fine_level]);
         dmem_all_data->output.prolong_wtime += MPI_Wtime() - begin;
         if (dmem_all_data->input.async_flag == 1){
            DMEM_AddCheckComm(dmem_all_data);
         }
         dmem_all_data->output.level_wtime[level] += MPI_Wtime() - level_begin;
      }
   }
   if (dmem_all_data->input.async_flag == 1){
      DMEM_AddCheckComm(dmem_all_data);
   }

  // if (my_grid == 0) hypre_ParVectorSetConstantValues(U_array[0], 0.0);

  // num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
  // u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
  // f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p && my_grid == 1){
  //       for (HYPRE_Int i = 0; i < num_rows; i++){
  //          printf("%e\n", f_local_data[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void DMEM_AddCorrect_LocalRes(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   double begin;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector **U_array;
   hypre_ParVector *e, *x;
   HYPRE_Real *e_local_data, *x_local_data, *u_local_data;
   HYPRE_Int num_rows;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   A_array = hypre_ParAMGDataAArray(amg_data);

   U_array = hypre_ParAMGDataUArray(amg_data);
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[finest_level]));

   e = dmem_all_data->vector_gridk.e;
   x = dmem_all_data->vector_gridk.x;
   hypre_ParVectorSetConstantValues(e, 0.0);
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   begin = MPI_Wtime();
   hypre_MPI_Waitall(dmem_all_data->comm.gridjToGridk_Correct_insideSend.procs.size(),
                     dmem_all_data->comm.gridjToGridk_Correct_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(),
                        dmem_all_data->comm.gridjToGridk_Correct_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }
   dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - begin;
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.gridjToGridk_Correct_insideSend),
            u_local_data,
            ACCUMULATE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.gridjToGridk_Correct_insideRecv),
            NULL,
            ACCUMULATE);

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend),
            u_local_data,
            ACCUMULATE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
            e_local_data,
            ACCUMULATE);

   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.gridjToGridk_Correct_insideRecv),
                e_local_data,
                ACCUMULATE);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                   e_local_data,
                   ACCUMULATE);
   }
   else {
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
               e_local_data,
               ACCUMULATE);
   }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;

   num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[finest_level]));
   begin = MPI_Wtime();
   for (HYPRE_Int i = 0; i < num_rows; i++){
      x_local_data[i] += e_local_data[i];
   }
   dmem_all_data->output.correct_wtime += MPI_Wtime() - begin;
   if (dmem_all_data->input.async_flag == 1){
      DMEM_AddCheckComm(dmem_all_data);
   }
}

void DMEM_AddResidual_LocalRes(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int my_grid = dmem_all_data->grid.my_grid;
   double begin;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   hypre_ParAMGData *amg_data_gridk =
      (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);
   hypre_ParCSRMatrix **A_array_gridk = hypre_ParAMGDataAArray(amg_data_gridk);

   begin = MPI_Wtime();
   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                      A_array_gridk[finest_level],
                                      dmem_all_data->vector_gridk.x,
                                      1.0,
                                      dmem_all_data->vector_gridk.b,
                                      F_array_gridk[finest_level]);
   dmem_all_data->output.residual_wtime += MPI_Wtime() - begin;
   if (dmem_all_data->input.async_flag == 1){
      DMEM_AddCheckComm(dmem_all_data);
   }
}

void DMEM_SolutionToFinest_LocalRes(DMEM_AllData *dmem_all_data,
                                    hypre_ParVector *x_gridk,
                                    hypre_ParVector *x_fine)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   int my_grid = dmem_all_data->grid.my_grid;
   double begin;

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   HYPRE_Real *x_gridk_local_data, *x_fine_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   x_gridk_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x_gridk));
   x_fine_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x_fine));

   begin = MPI_Wtime();
   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Correct_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - begin;

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Correct_insideSend),
            x_gridk_local_data,
            WRITE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
            NULL,
            READ);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                x_fine_local_data,
                READ);
}

void DMEM_VectorToGridk_LocalRes(DMEM_AllData *dmem_all_data,
                                   hypre_ParVector *r,
                                   hypre_ParVector *f)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   int my_grid = dmem_all_data->grid.my_grid;
   double begin;

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   HYPRE_Real *r_local_data, *f_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(r));
   f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(f));

   begin = MPI_Wtime();
   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Residual_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - begin;

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Residual_insideSend),
            r_local_data,
            WRITE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
            NULL,
            READ);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
                f_local_data,
                READ);
}

void DMEM_AddCorrect_GlobalRes(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   hypre_ParCSRMatrix **A_array_fine, **A_array_gridk;
   hypre_ParVector **U_array;
   hypre_ParVector *e, *x;
   HYPRE_Real *e_local_data, *x_local_data, *u_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   A_array_fine = hypre_ParAMGDataAArray(fine_amg_data);
   A_array_gridk = hypre_ParAMGDataAArray(gridk_amg_data);

   U_array = hypre_ParAMGDataUArray(gridk_amg_data);
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[finest_level]));

   e = dmem_all_data->vector_fine.e;
   x = dmem_all_data->vector_fine.x;
   hypre_ParVectorSetConstantValues(e, 0.0);
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

  // HYPRE_Int my_grid = dmem_all_data->grid.my_grid;
  // if (my_grid == 0){
  //    hypre_ParVectorSetConstantValues(U_array[0], 0.0);
  // }

   double begin = MPI_Wtime();
   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Correct_insideSend.requests,
                     MPI_STATUSES_IGNORE);
  // if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_outsideSend.procs.size(),
                        dmem_all_data->comm.finestToGridk_Correct_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
  // }

   SendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Correct_insideSend),
                 u_local_data,
                 ACCUMULATE);
   SendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Correct_outsideSend),
                 u_local_data,
                 ACCUMULATE);
   SendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                 NULL,
                 ACCUMULATE);
   SendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv),
                 e_local_data,
                 ACCUMULATE);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                e_local_data,
                ACCUMULATE);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv),
                   e_local_data,
                   ACCUMULATE);
   }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;

   fine_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_fine[finest_level]));
   for (HYPRE_Int i = 0; i < fine_num_rows; i++){
      x_local_data[i] += e_local_data[i];
   }
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       for (HYPRE_Int i = 0; i < fine_num_rows; i++){
  //          printf("%d %d %e\n", my_id, fine_first_row+i, x_local_data[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void DMEM_AddResidual_GlobalRes(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int my_grid = dmem_all_data->grid.my_grid;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData*)dmem_all_data->hypre.solver;

   hypre_ParCSRMatrix *A = dmem_all_data->matrix.A_fine;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);

   hypre_ParVector *x = dmem_all_data->vector_fine.x;
   hypre_ParVector *b = dmem_all_data->vector_fine.b;
   hypre_ParVector *r = dmem_all_data->vector_fine.r;
   hypre_Vector *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector *b_local  = hypre_ParVectorLocalVector(b);
   hypre_Vector *r_local  = hypre_ParVectorLocalVector(r);
   hypre_Vector *x_ghost = dmem_all_data->vector_fine.x_ghost;

  // hypre_Vector *x_ghost = hypre_SeqVectorCreate(num_cols_offd);
  // hypre_SeqVectorInitialize(x_ghost);
   
   HYPRE_Real *x_local_data  = hypre_VectorData(x_local);
   HYPRE_Real *x_ghost_data  = hypre_VectorData(x_ghost);

   
   double begin = MPI_Wtime();
   hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_insideSend.procs.size(),
                     dmem_all_data->comm.finestIntra_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   SendRecv(dmem_all_data,
                &(dmem_all_data->comm.finestIntra_insideSend),
                x_local_data,
                WRITE);
  // if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                        dmem_all_data->comm.finestIntra_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
  // }
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestIntra_outsideSend),
            x_local_data,
            WRITE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestIntra_insideRecv),
            NULL,
            READ);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestIntra_outsideRecv),
            x_ghost_data,
            READ);

   hypre_CSRMatrixMatvecOutOfPlace(-1.0,
                                   diag,
                                   x_local,
                                   1.0,
                                   b_local,
                                   r_local,
                                   0);

   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestIntra_insideRecv),
                x_ghost_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.finestIntra_outsideRecv),
                   x_ghost_data,
                   READ);
   }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;

   hypre_CSRMatrixMatvec(-1.0,
                         offd,
                         x_ghost,
                         1.0,
                         r_local);

   HYPRE_Real *r_local_data = hypre_VectorData(r_local);
   amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[finest_level]));

   begin = MPI_Wtime();
   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Residual_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   SendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Residual_insideSend),
                 r_local_data,
                 WRITE);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_outsideSend.procs.size(),
                        dmem_all_data->comm.finestToGridk_Residual_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Residual_outsideSend),
            r_local_data,
            WRITE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
            NULL,
            READ);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv),
            f_local_data,
            READ); 
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
                f_local_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv),
                   f_local_data,
                   READ);
   }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
  // hypre_SeqVectorDestroy(x_ghost);
}

void FineSmooth(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData*)dmem_all_data->hypre.solver;

   hypre_ParCSRMatrix *A = dmem_all_data->matrix.A_fine;
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);

   hypre_ParVector *x = dmem_all_data->vector_fine.x;
   hypre_ParVector *r = dmem_all_data->vector_fine.r;
   hypre_Vector *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector *r_local  = hypre_ParVectorLocalVector(r);
 
   HYPRE_Real *x_local_data  = hypre_VectorData(x_local);
   HYPRE_Real *r_local_data  = hypre_VectorData(r_local);

   HYPRE_Real *A_data = hypre_CSRMatrixData(diag);
   HYPRE_Int *A_i = hypre_CSRMatrixI(diag);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(diag);
   for (HYPRE_Int i = 0; i < num_rows; i++){
      x_local_data[i] += dmem_all_data->input.smooth_weight * r_local_data[i] / A_data[A_i[i]];
   }
}

void PrintMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
//      if (comm_data->message_count[i] < dmem_all_data->input.num_inner_cycles)
      printf("%d, %d, %d\n", comm_data->type, comm_data->message_count[i], dmem_all_data->input.num_inner_cycles);
   }
}

int DMEM_CheckMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      if (comm_data->message_count[i] < dmem_all_data->input.num_inner_cycles){
         return 0;
      }
   }
   return 1;
}

void PrintMessageFlags(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      printf("%d\n", comm_data->done_flags[i]);
   }
}

int DMEM_CheckMessageFlagsValue(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int value)
{
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      if (comm_data->done_flags[i] != value){
         return 0;
      }
   }
   return 1;
}

int DMEM_CheckMessageFlagsNotValue(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int value)
{
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      if (comm_data->done_flags[i] == value){
         return 0;
      }
   }
   return 1;
}

int DMEM_CheckOutsideDoneFlag(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      if (DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideSend), 0) == 1 &&
          DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv), 0) == 1 &&
          DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideSend), 0) == 1 &&
          DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv), 0) == 1 &&
          DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend), 0) == 1 &&
          DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv), 0) == 1){
         dmem_all_data->comm.all_done_flag = 1;
         return 1;
      }
   }
   else {
      if (DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), 0) == 1 &&
          DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv), 0) == 1){
         dmem_all_data->comm.outside_done_flag = 1;
         return 1;
      }
   }
   return 0;
}

void DMEM_AsyncRecvStart(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   MPI_Status status;
   int flag;
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      HYPRE_Int ip = comm_data->procs[i];
      HYPRE_Int vec_len = comm_data->len[i];
      hypre_MPI_Irecv(comm_data->data[i],
                      vec_len+1,
                      HYPRE_MPI_REAL,
                      ip,
                      comm_data->tag,
                      MPI_COMM_WORLD,
                      &(comm_data->requests[i]));
   }
}

void AsyncStart(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv));
      DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv));
      DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));
   }
   else {
      DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv));
   }
}

void AsyncRecvCleanup(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector *e, *x;
   HYPRE_Real *e_local_data, *x_local_data;
   HYPRE_Int num_rows;
   double begin;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      e = dmem_all_data->vector_fine.e;
      x = dmem_all_data->vector_fine.x;
      amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
      A_array = hypre_ParAMGDataAArray(amg_data);
      num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[finest_level]));
   }
   else {
      e = dmem_all_data->vector_gridk.e;
      x = dmem_all_data->vector_gridk.x;
      amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
      A_array = hypre_ParAMGDataAArray(amg_data);
      num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[finest_level]));
   }

   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

  // PrintMessageFlags(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend));

   while (1){
      int break_flag = 0;
      if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
         if (DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv), 2) == 1 &&
             DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv), 2) == 1 &&
             DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv), 2) == 1){
            break;
         }
      }
      else {
         if (DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv), 2) == 1){
            break;
         }
      }

      hypre_ParVectorSetConstantValues(e, 0.0);
      begin = MPI_Wtime();
      if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
         SendRecv(dmem_all_data,
                  &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv),
                  e_local_data,
                  ACCUMULATE);
      }
      else {
         SendRecv(dmem_all_data,
                  &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                  e_local_data,
                  ACCUMULATE);
      }
      dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
      for (HYPRE_Int i = 0; i < num_rows; i++){
         x_local_data[i] += e_local_data[i];
      }

      begin = MPI_Wtime();
      if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
         SendRecv(dmem_all_data,
                  &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv),
                  NULL,
                  -1);
         SendRecv(dmem_all_data,
                  &(dmem_all_data->comm.finestIntra_outsideRecv),
                  NULL,
                  -1);
      }
      dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
   }

   begin = MPI_Wtime();
   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_outsideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Correct_outsideSend.requests,
                     MPI_STATUSES_IGNORE);
      hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_outsideSend.procs.size(),
                        dmem_all_data->comm.finestToGridk_Residual_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
      hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                        dmem_all_data->comm.finestIntra_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }
   else {
      CompleteInFlight(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend));
   }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
}

void AsyncEnd(DMEM_AllData *dmem_all_data)
{
   AsyncRecvCleanup(dmem_all_data);
}

void MPITestComm(DMEM_AllData *dmem_all_data,
                 DMEM_CommData *comm_data)
{
   HYPRE_Int flag;
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   double begin;
   
   begin = MPI_Wtime();
   for (int i = 0; i < comm_data->procs.size(); i++){
      hypre_MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
   }
   dmem_all_data->output.residual_wtime += MPI_Wtime() - begin;
}

void DMEM_AddCheckComm(DMEM_AllData *dmem_all_data)
{
   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector *e;
   hypre_ParVector *x;
   HYPRE_Real *e_local_data;
   HYPRE_Real *x_local_data;
   int flag;
   double begin;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
      A_array = hypre_ParAMGDataAArray(amg_data);

      e = dmem_all_data->vector_fine.e;
      x = dmem_all_data->vector_fine.x;
      hypre_ParVectorSetConstantValues(e, 0.0);
      e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
      x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
      begin = MPI_Wtime();
      SendRecv(dmem_all_data,
                    &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv),
                    e_local_data,
                    ACCUMULATE);
      dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
      HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[finest_level]));
      for (HYPRE_Int i = 0; i < num_rows; i++){
         x_local_data[i] += e_local_data[i];
      }

      hypre_Vector *x_ghost = dmem_all_data->vector_fine.x_ghost;
      HYPRE_Real *x_ghost_data = hypre_VectorData(x_ghost);
      begin = MPI_Wtime();
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.finestIntra_outsideRecv),
               x_ghost_data,
               READ);
      dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;

      amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
      hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
      HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[finest_level]));
      begin = MPI_Wtime();
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv),
               f_local_data,
               READ);
      dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
   }
   else {
      amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
      A_array = hypre_ParAMGDataAArray(amg_data);
                       
      e = dmem_all_data->vector_gridk.e;
      x = dmem_all_data->vector_gridk.x;
      hypre_ParVectorSetConstantValues(e, 0.0);
      e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
      x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

      begin = MPI_Wtime();
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
               e_local_data,
               ACCUMULATE);
      dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
      HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[finest_level]));
      for (HYPRE_Int i = 0; i < num_rows; i++){
         x_local_data[i] += e_local_data[i];
      }
      begin = MPI_Wtime();
      for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(); i++){
         CheckInFlight(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), i);
      }
      dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
   }
}

int CheckConverge(DMEM_AllData *dmem_all_data)
{
   double begin;
   int cycle, num_cycles;

   if (dmem_all_data->input.solver == MULT_MULTADD){
      cycle = dmem_all_data->iter.inner_cycle;
      num_cycles = dmem_all_data->input.num_inner_cycles;
   }
   else {
      cycle = dmem_all_data->iter.cycle;
      num_cycles = dmem_all_data->input.num_cycles;
   }

   if (dmem_all_data->input.async_flag == 1){
      if (dmem_all_data->comm.all_done_flag == 1){
         if (DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), 2) == 1){ 
            return 1;
         }
      }
      else {
         if (dmem_all_data->input.converge_test_type == GLOBAL_CONVERGE){
            int my_comm_done;
            DMEM_CheckOutsideDoneFlag(dmem_all_data);
            begin = MPI_Wtime();
            MPI_Allreduce(&(dmem_all_data->comm.outside_done_flag),
                          &(my_comm_done),
                          1,
                          MPI_INT,
                          MPI_MIN,
                          dmem_all_data->grid.my_comm);
            dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
            if (my_comm_done == 1){
               dmem_all_data->comm.all_done_flag = 1;
              // return 1;
            }
         }
         else {
            if (cycle >= num_cycles-1 || dmem_all_data->iter.r_norm2_local_converge_flag == 1){
               dmem_all_data->comm.all_done_flag = 1;
              // return 1;
            }
           // if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
           //    if (CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideSend), 2) == 1 &&
           //        CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideSend), 2) == 1 &&
           //        CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend), 2) == 1){
           //       PrintMessageFlags(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideSend));
           //       PrintMessageFlags(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideSend));
           //       PrintMessageFlags(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend));
           //       return 1;
           //    }
           // }
           // else {
           //    if (CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), 2) == 1){
           //       return 1;
           //    }
           // }
         }
      }
   }
   else {
      if (cycle >= num_cycles-1 || dmem_all_data->iter.r_norm2_local_converge_flag == 1){
         return 1;
      }
   }
   
   return 0;
}
