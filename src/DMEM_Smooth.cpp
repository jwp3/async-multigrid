#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Test.hpp"
#include "DMEM_Smooth.hpp"
#include "DMEM_Add.hpp"

int AsyncSmoothCheckConverge(DMEM_AllData *dmem_all_data);
void AsyncSmoothAddCorrect_LocalRes(DMEM_AllData *dmem_all_data);
void AsyncSmoothRecvCleanup(DMEM_AllData *dmem_all_data);
void AsyncSmoothEnd(DMEM_AllData *dmem_all_data);
double StochasticParallelSouthwellUpdateProbability(DMEM_AllData *dmem_all_data);
int AsyncSmoothCheckComm(DMEM_AllData *dmem_all_data);

void DMEM_AsyncSmooth(DMEM_AllData *dmem_all_data, int level)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int recv_flag;
   double comp_begin, residual_begin, comm_begin, residual_norm_begin;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
 
   hypre_ParCSRMatrix *A = A_array[level]; 

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   HYPRE_Real *b_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.b));
   HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));
   HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x));
   HYPRE_Real *r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r));
  // HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.e));
  // hypre_ParVectorSetConstantValues(dmem_all_data->vector_gridk.e, 0.0);

   HYPRE_Real *x_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost);
   HYPRE_Real *x_ghost_prev_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost_prev);
  // HYPRE_Real *b_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.b_ghost);
  // HYPRE_Real *a_diag_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.a_diag_ghost);

  // double begin = MPI_Wtime();
   if (dmem_all_data->input.async_flag == 1){
      dmem_all_data->comm.outside_recv_done_flag = 0;
      dmem_all_data->comm.is_async_smoothing_flag = 1;
      dmem_all_data->comm.async_smooth_done_flag = 0;
      DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));
   }

  // if (dmem_all_data->input.res_update_type == RES_ACCUMULATE ||
  //     dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
  //    dmem_all_data->comm.finestIntra_outsideRecv.update_res_in_comm = 1;
  // }

   if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
      for (int i = 0; i < dmem_all_data->comm.finestIntra_outsideRecv.procs.size(); i++){
         dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary[i] = 0.0;
         for (int j = 0; j < num_rows; j++){
            for (int k = 0; k < dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_j[i][j].size(); k++){
               int ii = dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_j[i][j][k];
               HYPRE_Real aij = dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_data[i][j][k];
               dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary[i] += fabs(aij * r_local_data[j]);
               x_ghost_prev_data[ii] = x_ghost_data[ii];
            }
         }
         dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary_prev[i] = dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary[i];
      }
   }

   int converge_flag = 0; 
   int update_flag = 1;
   while (1){
      converge_flag = AsyncSmoothCheckConverge(dmem_all_data);
      if (converge_flag == 1){
         update_flag = 1;
      }
      if (update_flag == 1){
         comp_begin = MPI_Wtime();
         if (dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL || dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
            comp_begin = MPI_Wtime();
            for (int i = 0; i < num_rows; i++){
               u_local_data[i] = 0.0;
            }
            for (int i = 0; i < num_rows; i++){
               HYPRE_Real res = b_local_data[i];
               for (int jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++){
                  int ii = A_diag_j[jj];
                  res -= A_diag_data[jj] * x_local_data[ii];
               }
               for (int jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++){
                  int ii = A_offd_j[jj];
                  res -= A_offd_data[jj] * x_ghost_data[ii];
               }
               u_local_data[i] = res / A_diag_data[A_diag_i[i]];
               x_local_data[i] += u_local_data[i];
               r_local_data[i] -= u_local_data[i];
            }
            dmem_all_data->output.comp_wtime += MPI_Wtime() - comp_begin;
         }
         else {
            comp_begin = MPI_Wtime();
           // for (int i = 0; i < num_rows; i++){
           //    u_local_data[i] = dmem_all_data->input.smooth_weight * r_local_data[i] / A_diag_data[A_diag_i[i]];
           // }
            DMEM_HypreParVector_Set(U_array[0], 0.0, num_rows);
            DMEM_HypreParVector_VecAxpy(U_array[0], dmem_all_data->vector_gridk.r, dmem_all_data->matrix.wJacobi_scale_gridk[0], num_rows);

           // for (int i = 0; i < num_rows; i++){
           //    HYPRE_Real Au_i = 0.0;
           //    for (int jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++){
           //       int ii = A_diag_j[jj];
           //       Au_i += A_diag_data[jj] * u_local_data[ii];
           //    }
           //    for (int jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++){
           //       int ii = A_offd_j[jj];
           //       HYPRE_Real u_ghost_i = b_ghost_data[ii] - A_offd_data[jj] * x_ghost_data[ii]) / a_diag_ghost_data[ii];
           //       Au_i += dmem_all_data->input.smooth_weight * (b_ghost_data[ii] - A_offd_data[jj] * x_ghost_data[ii]) / a_diag_ghost_data[ii];
           //    }
           //    u_local_data[i] = 2.0 * u_local_data[i] -  dmem_all_data->input.smooth_weight * Au_i / A_diag_data[A_diag_i[i]];
           // }
            dmem_all_data->output.comp_wtime += MPI_Wtime() - comp_begin;
         }
         dmem_all_data->output.comp_wtime += MPI_Wtime() - comp_begin;
         dmem_all_data->iter.relax += 1;

         AsyncSmoothAddCorrect_LocalRes(dmem_all_data);
         if (dmem_all_data->input.smoother == ASYNC_JACOBI){
            comp_begin = MPI_Wtime();
            DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, U_array[0], 1.0, num_rows);
           // for (int i = 0; i < num_rows; i++){
           //    x_local_data[i] += u_local_data[i];
           // }
            dmem_all_data->output.comp_wtime += MPI_Wtime() - comp_begin;
         }
      }
      comm_begin = MPI_Wtime();
      if (dmem_all_data->input.smoother != ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
         if (dmem_all_data->input.res_update_type == RES_ACCUMULATE){
            SendRecv(dmem_all_data,
                     &(dmem_all_data->comm.finestIntra_outsideSend),
                     u_local_data,
                     ACCUMULATE);
         }
         else {
            SendRecv(dmem_all_data,
                     &(dmem_all_data->comm.finestIntra_outsideSend),
                     x_local_data,
                     WRITE);
         }
      }
      recv_flag = SendRecv(dmem_all_data,
                           &(dmem_all_data->comm.finestIntra_outsideRecv),
                           x_ghost_data,
                           READ); 
      dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
      if (dmem_all_data->input.async_flag == 0 &&
          dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL){
         comm_begin = MPI_Wtime();
         CompleteRecv(dmem_all_data,
                      &(dmem_all_data->comm.finestIntra_outsideRecv),
                      x_ghost_data,
                      READ);
         double mpiwait_begin = MPI_Wtime();
         hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                           dmem_all_data->comm.finestIntra_outsideSend.requests,
                           MPI_STATUSES_IGNORE);
         dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - mpiwait_begin;
         dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
      }
      
      residual_begin = MPI_Wtime();
      if (dmem_all_data->input.smoother == ASYNC_JACOBI){
         if (dmem_all_data->input.res_update_type == RES_ACCUMULATE){
            if (update_flag == 1){
               for (int i = 0; i < num_rows; i++){
                  for (int jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++){
                     int ii = A_diag_j[jj];
                     r_local_data[i] -= A_diag_data[jj] * u_local_data[ii];
                  }
               }
              // if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
              //    for (int i = 0; i < dmem_all_data->comm.finestIntra_outsideRecv.procs.size(); i++){
              //       if (dmem_all_data->comm.finestIntra_outsideRecv.message_count[i] > 0){
              //          dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary[i] = 0.0;
              //          for (int j = 0; j < num_rows; j++){
              //             for (int k = 0; k < dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_j[i][j].size(); k++){
              //                int ii = dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_j[i][j][k];
              //                HYPRE_Real aij = dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_data[i][j][k];
              //                dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary[i] += fabs(aij * r_local_data[j]);
              //             }
              //          }
              //          dmem_all_data->comm.finestIntra_outsideRecv.r_norm[i] += dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary[i] - dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary_prev[i];
              //          dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary_prev[i] = dmem_all_data->comm.finestIntra_outsideRecv.r_norm_boundary[i];
              //       }
              //    }
              // }
            }
            
            if (dmem_all_data->input.async_flag == 0){ 
               comm_begin = MPI_Wtime();
               if (dmem_all_data->input.async_flag == 0){
                  CompleteRecv(dmem_all_data,
                               &(dmem_all_data->comm.finestIntra_outsideRecv),
                               x_ghost_data,
                               READ);
                  double mpiwait_begin = MPI_Wtime();
                  hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                                    dmem_all_data->comm.finestIntra_outsideSend.requests,
                                    MPI_STATUSES_IGNORE);
                  dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - mpiwait_begin;
               }
               dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
               comp_begin = MPI_Wtime();
               for (int i = 0; i < num_rows; i++){
                  for (int jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++){
                     int ii = A_offd_j[jj];
                     r_local_data[i] -= A_offd_data[jj] * x_ghost_data[ii];
                  }
               }
               dmem_all_data->output.comp_wtime += MPI_Wtime() - comp_begin;
            }
         }
         else {
            comp_begin = MPI_Wtime();
           // for (int i = 0; i < num_rows; i++){
           //    r_local_data[i] = b_local_data[i];
           //    for (int jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++){
           //       int ii = A_diag_j[jj];
           //       r_local_data[i] -= A_diag_data[jj] * x_local_data[ii];
           //    }
           // }
            hypre_CSRMatrixMatvecOutOfPlace(-1.0,
                                            A_diag,
                                            hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x),
                                            1.0,
                                            hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.b),
                                            hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r),
                                            0);
            dmem_all_data->output.comp_wtime += MPI_Wtime() - comp_begin;

            if (dmem_all_data->input.async_flag == 0){
               comm_begin = MPI_Wtime();
               if (dmem_all_data->input.async_flag == 0){
                  CompleteRecv(dmem_all_data,
                               &(dmem_all_data->comm.finestIntra_outsideRecv),
                               x_ghost_data,
                               READ);
                  double mpiwait_begin = MPI_Wtime();
                  hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                                    dmem_all_data->comm.finestIntra_outsideSend.requests,
                                    MPI_STATUSES_IGNORE);
                  dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - mpiwait_begin;
               }
               dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin; 
            }
            comp_begin = MPI_Wtime();
           // for (int i = 0; i < num_rows; i++){
           //    for (int jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++){
           //       int ii = A_offd_j[jj];
           //       r_local_data[i] -= A_offd_data[jj] * x_ghost_data[ii];
           //    }
           // }
            DMEM_HypreParVector_Copy(Vtemp, dmem_all_data->vector_gridk.r, num_rows);
            hypre_CSRMatrixMatvecOutOfPlace(-1.0,
                                            A_offd,
                                            dmem_all_data->vector_gridk.x_ghost,
                                            1.0,
                                            hypre_ParVectorLocalVector(Vtemp),
                                            hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r),
                                            0);
            dmem_all_data->output.comp_wtime += MPI_Wtime() - comp_begin;
         }
      }
      dmem_all_data->output.residual_wtime += MPI_Wtime() - residual_begin;
      
      if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
         int recv_flag2 = AsyncSmoothCheckComm(dmem_all_data);
         if (update_flag == 1 || recv_flag2 == 1){
            comp_begin = MPI_Wtime();
            for (int i = 0; i < num_rows; i++){
               r_local_data[i] = b_local_data[i];
               for (int jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++){
                  int ii = A_diag_j[jj];
                  r_local_data[i] -= A_diag_data[jj] * x_local_data[ii];
               }
               for (int jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++){
                  int ii = A_offd_j[jj];
                  r_local_data[i] -= A_offd_data[jj] * x_ghost_data[ii];
               }
            }
            residual_norm_begin = MPI_Wtime();
            dmem_all_data->iter.r_L1norm_local = 0.0;
            for (int i = 0; i < num_rows; i++){
               dmem_all_data->iter.r_L1norm_local += fabs(r_local_data[i]);
            }
            dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
           // printf("%d %d %e\n", my_id, dmem_all_data->iter.cycle, dmem_all_data->iter.r_L1norm_local);
            dmem_all_data->output.comp_wtime += MPI_Wtime() - residual_norm_begin;

            comm_begin = MPI_Wtime();
            if (dmem_all_data->input.res_update_type == RES_ACCUMULATE){
               SendRecv(dmem_all_data,
                        &(dmem_all_data->comm.finestIntra_outsideSend),
                        u_local_data,
                        ACCUMULATE);
            }
            else {
               SendRecv(dmem_all_data,
                        &(dmem_all_data->comm.finestIntra_outsideSend),
                        x_local_data,
                        WRITE);
            }
            dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
         }

         double update_probability = StochasticParallelSouthwellUpdateProbability(dmem_all_data); 
         update_flag = 0;
         if (RandDouble(0.0, 1.0) < update_probability){
            update_flag = 1;
         }
      }


      if (dmem_all_data->input.async_flag == 0){
         residual_norm_begin = MPI_Wtime();
         hypre_Vector *r = hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r);
         dmem_all_data->iter.r_L2norm_local =
            sqrt(InnerProd(r, r, dmem_all_data->grid.my_comm))/dmem_all_data->output.r0_norm2;
         dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
         if (dmem_all_data->iter.r_L2norm_local < dmem_all_data->input.tol){
            dmem_all_data->iter.r_L2norm_local_converge_flag = 1;
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
   
   double end_begin = MPI_Wtime();
   if (dmem_all_data->input.async_flag == 1){
      AsyncSmoothEnd(dmem_all_data);
   }
   //printf("%d %d\n", my_id, dmem_all_data->iter.relax);
   dmem_all_data->output.end_wtime += MPI_Wtime() - end_begin;
  // printf("%d %e\n", my_id, MPI_Wtime()-begin);
  // hypre_ParVectorCopy(dmem_all_data->vector_gridk.r, F_array[level]);
   dmem_all_data->comm.is_async_smoothing_flag = 0;
}

int AsyncSmoothCheckConverge(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->input.async_flag == 1){
      if (dmem_all_data->comm.async_smooth_done_flag == 1){
         if (DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), 2) == 1 &&
             DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend), 2) == 1){
            return 1;
         }
      }
      else {
         if (dmem_all_data->comm.outside_recv_done_flag == 1){
            if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
               dmem_all_data->comm.async_smooth_done_flag = 1;
            }
            else {
               if (DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend), 0) == 1 &&
                   DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv), 0) == 1){
                  dmem_all_data->comm.async_smooth_done_flag = 1;
               }
            }
         }
         else {
            if (DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv), 0)){
               dmem_all_data->comm.outside_recv_done_flag = 1;
            }
         }
      }
   }
   else {
      int cycle, num_cycles;
      if (dmem_all_data->input.solver == MULT_MULTADD){
         cycle = dmem_all_data->iter.inner_cycle;
         num_cycles = dmem_all_data->input.num_inner_cycles;
      }
      else {
         cycle = dmem_all_data->iter.cycle;
         num_cycles = dmem_all_data->input.num_cycles;
      }

      if (cycle >= num_cycles-1 || dmem_all_data->iter.r_L2norm_local_converge_flag == 1){
         return 1;
      }
   }
   return 0;
}

void AsyncSmoothAddCorrect_LocalRes(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   double begin;
   int recv_flag;

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
   num_rows = hypre_ParCSRMatrixNumRows(A_array[finest_level]);

   e = dmem_all_data->vector_gridk.e;
   x = dmem_all_data->vector_gridk.x;
   DMEM_HypreParVector_Set(e, 0.0, num_rows);
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[finest_level]));
   
   begin = MPI_Wtime();
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(),
                        dmem_all_data->comm.gridjToGridk_Correct_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
      dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - begin;
   }

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend),
            u_local_data,
            ACCUMULATE);
   recv_flag = SendRecv(dmem_all_data,
                        &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                        e_local_data,
                        ACCUMULATE);

   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                   e_local_data,
                   ACCUMULATE);
      recv_flag = 1;
   }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;

   begin = MPI_Wtime();
   if (recv_flag == 1){
      if (dmem_all_data->input.res_update_type == RES_ACCUMULATE){
         DMEM_HypreParVector_Axpy(U_array[0], dmem_all_data->vector_gridk.e, 1.0, num_rows);
      }
      else {
         DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, dmem_all_data->vector_gridk.e, 1.0, num_rows);
      }
   }
   dmem_all_data->output.correct_wtime += MPI_Wtime() - begin;
   dmem_all_data->output.comp_wtime += MPI_Wtime() - begin;
}

int AsyncSmoothCheckComm(DMEM_AllData *dmem_all_data)
{
   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector **U_array;
   hypre_ParVector *e;
   hypre_ParVector *x;
   HYPRE_Real *e_local_data, *x_local_data, *u_local_data;
   int flag;
   double begin;
   
   int finest_level = dmem_all_data->input.coarsest_mult_level;
 
   int recv_flag;
   amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   A_array = hypre_ParAMGDataAArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[finest_level]);


   e = dmem_all_data->vector_gridk.e;
   x = dmem_all_data->vector_gridk.x;
   DMEM_HypreParVector_Set(e, 0.0, num_rows);
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[finest_level]));

   begin = MPI_Wtime();
   recv_flag = SendRecv(dmem_all_data,
                        &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                        e_local_data,
                        ACCUMULATE);
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
   if (recv_flag == 1){
      if (dmem_all_data->input.res_update_type == RES_ACCUMULATE){
         DMEM_HypreParVector_Axpy(U_array[0], dmem_all_data->vector_gridk.e, 1.0, num_rows);
      }
      else {
         DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, dmem_all_data->vector_gridk.e, 1.0, num_rows);
      }
   }
   begin = MPI_Wtime();
   for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(); i++){
      CheckInFlight(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), i);
   }
   for (int i = 0; i < dmem_all_data->comm.finestIntra_outsideSend.procs.size(); i++){
      CheckInFlight(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend), i);
   }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
   return recv_flag;
}

void AsyncSmoothRecvCleanup(DMEM_AllData *dmem_all_data)
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
   e = dmem_all_data->vector_gridk.e;
   x = dmem_all_data->vector_gridk.x;
   amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   A_array = hypre_ParAMGDataAArray(amg_data);
   num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[finest_level]));

   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   HYPRE_Real *x_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost);

   while (1){
      if (DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv), 2) == 1){
         break;
      }

      DMEM_HypreParVector_Set(e, 0.0, num_rows);
      begin = MPI_Wtime();
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
               e_local_data,
               ACCUMULATE);
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.finestIntra_outsideRecv),
               x_ghost_data,
               READ);      
      dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
      begin = MPI_Wtime();
      DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, dmem_all_data->vector_gridk.e, 1.0, num_rows);
      dmem_all_data->output.comp_wtime += MPI_Wtime() - begin;
   }


   begin = MPI_Wtime();
   CompleteInFlight(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend));
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;

  // if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
  //    int num_procs;
  //    MPI_Comm_size(dmem_all_data->grid.my_comm, &num_procs);
  //    MPI_Comm_rank(dmem_all_data->grid.my_comm, &my_id);
  //    MPI_Barrier(dmem_all_data->grid.my_comm);
  //    for (int p = 0; p < num_procs; p++){
  //       if (p == my_id){
  //          printf("%d %e:\n\t", my_id, dmem_all_data->iter.r_L1norm_local);
  //          for (int i = 0; i < dmem_all_data->comm.finestIntra_outsideRecv.procs.size(); i++){
  //             printf(" (%d, %e)", dmem_all_data->comm.finestIntra_outsideRecv.procs[i], dmem_all_data->comm.finestIntra_outsideRecv.r_norm[i]);
  //          }
  //          printf("\n");
  //       }
  //       MPI_Barrier(dmem_all_data->grid.my_comm);
  //    }
  // }
}

void AsyncSmoothEnd(DMEM_AllData *dmem_all_data)
{
   for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_insideRecv.procs.size(); i++){
      dmem_all_data->comm.gridjToGridk_Correct_insideRecv.done_flags[i] = 2;
   }
   for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_insideSend.procs.size(); i++){
      dmem_all_data->comm.gridjToGridk_Correct_insideSend.done_flags[i] = 2;
   }
   AsyncSmoothRecvCleanup(dmem_all_data);
}

double StochasticParallelSouthwellUpdateProbability(DMEM_AllData *dmem_all_data)
{
   double x = 0.0;
   for (int i = 0; i < dmem_all_data->comm.finestIntra_outsideRecv.procs.size(); i++){
      if (dmem_all_data->iter.r_L1norm_local < dmem_all_data->comm.finestIntra_outsideRecv.r_norm[i]){
         x++;
      }
   }
   
   double p;
   double alpha = dmem_all_data->input.sps_alpha;
   if (dmem_all_data->input.sps_probability_type == SPS_PROBABILITY_INVERSE){ 
      p = (1.0/x)*(1.0/alpha);
   }
   else {
      p = exp(-x*alpha);
   }

   return p;
}

void DMEM_AddSmooth(DMEM_AllData *dmem_all_data, int coarsest_level)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   HYPRE_Int num_rows;
   int fine_level, coarse_level;

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
   hypre_ParVector *e = dmem_all_data->vector_gridk.e;

   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   int num_levels = dmem_all_data->grid.num_levels;
   int my_grid = dmem_all_data->grid.my_grid;

   HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
   HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
   num_rows = hypre_ParCSRMatrixNumRows(A_array[coarsest_level]);
   f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[coarsest_level]));
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[coarsest_level]));

   double begin = MPI_Wtime(); 
   if (dmem_all_data->input.solver == BPX){
      double prob = RandDouble(0, 1.0);
      for (int i = 0; i < num_rows; i++){
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
      DMEM_HypreParVector_Set(U_array[coarsest_level], 0.0, num_rows);
      if (dmem_all_data->input.smoother == L1_JACOBI){
         DMEM_HypreParVector_VecAxpy(U_array[coarsest_level], F_array[coarsest_level], dmem_all_data->matrix.L1_row_norm_gridk[coarsest_level], num_rows);
      }
      else {
         for (int i = 0; i < num_rows; i++){
            u_local_data[i] = dmem_all_data->input.smooth_weight * f_local_data[i] / A_data[A_i[i]];
         }
        // DMEM_HypreParVector_VecAxpy(U_array[coarsest_level], F_array[coarsest_level], dmem_all_data->matrix.wJacobi_scale_gridk[coarsest_level], num_rows);
      }
      hypre_ParCSRMatrixMatvec(1.0,
                               A_array[coarsest_level],
                               U_array[coarsest_level],
                               0.0,
                               Vtemp);

     // hypre_ParCSRMatrixMatvecOutOfPlace(1.0,
     //                                    A_array[coarsest_level],
     //                                    U_array[coarsest_level],
     //                                    0.0,
     //                                    e,
     //                                    Vtemp);
      if (dmem_all_data->input.smoother == L1_JACOBI){
         DMEM_HypreParVector_VecAxpy(U_array[coarsest_level], Vtemp, dmem_all_data->matrix.symmL1_row_norm_gridk[coarsest_level], num_rows);
      }
      else {
         for (int i = 0; i < num_rows; i++){
            u_local_data[i] = 2.0 * u_local_data[i] - dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
         }
        // DMEM_HypreParVector_VecAxpy(U_array[coarsest_level], Vtemp, dmem_all_data->matrix.symmwJacobi_scale_gridk[coarsest_level], num_rows);
      }
     // DMEM_HypreParVector_Scale(U_array[coarsest_level], 2.0, num_rows);
   }
   //dmem_all_data->output.comp_wtime += MPI_Wtime() - begin;
}
