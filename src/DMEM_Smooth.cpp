#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Test.hpp"
#include "DMEM_Smooth.hpp"
#include "DMEM_Add.hpp"

int AsyncSmoothCheckConverge(DMEM_AllData *dmem_all_data);
int AsyncSmoothAddCorrect_LocalRes(DMEM_AllData *dmem_all_data);
void AsyncSmoothRecvCleanup(DMEM_AllData *dmem_all_data);
double StochasticParallelSouthwellUpdateProbability(DMEM_AllData *dmem_all_data);
int AsyncSmoothCheckComm(DMEM_AllData *dmem_all_data);

void DMEM_AsyncSmooth(DMEM_AllData *dmem_all_data, int level)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int recv_gridk_flag, recv_intra_flag;
   int send_gridk_flag = 1, send_intra_flag = 1;
   double matvec_begin, vecop_begin, residual_begin, residual_norm_begin, comm_begin, mpiwait_begin;
   double matvec_end, vecop_end, residual_end, comm_end, mpiwait_end, residual_norm_end;

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
   HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.e));
   HYPRE_Real *v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   HYPRE_Real *x_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost);
   HYPRE_Real *x_ghost_prev_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost_prev);

   vecop_begin = MPI_Wtime();
   DMEM_HypreParVector_Set(Vtemp, 0.0, num_rows);
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

   dmem_all_data->comm.outside_recv_done_flag = 0;
   dmem_all_data->comm.is_async_smoothing_flag = 1;
   dmem_all_data->comm.async_smooth_done_flag = 0;
   DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));


   // TODO: fix Gauss-Seidel solvers
   int converge_flag = 0; 
   int update_flag = 1;
   while (1){
      converge_flag = AsyncSmoothCheckConverge(dmem_all_data);
      if (converge_flag == 1){
         update_flag = 1;
      }
      if (update_flag == 1){
         if (dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL ||
             dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL){
            matvec_begin = MPI_Wtime();
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
               u_local_data[i] = dmem_all_data->input.smooth_weight * res / A_diag_data[A_diag_i[i]];
               x_local_data[i] += u_local_data[i];
               r_local_data[i] -= u_local_data[i];
            }
            dmem_all_data->output.matvec_wtime += MPI_Wtime() - matvec_begin;
         }
         else {
            vecop_begin = MPI_Wtime();
            DMEM_HypreParVector_Set(U_array[level], 0.0, num_rows);
            if (dmem_all_data->input.smoother == ASYNC_L1_JACOBI){
               DMEM_HypreParVector_Ivaxpy(U_array[level], dmem_all_data->vector_gridk.r, dmem_all_data->matrix.L1_row_norm_gridk[level], num_rows);
            }
            else {
               DMEM_HypreParVector_Ivaxpy(U_array[level], dmem_all_data->vector_gridk.r, dmem_all_data->matrix.wJacobi_scale_gridk[level], num_rows);
            }
            dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
         }
         if (dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL){
            DMEM_ChebyUpdate(dmem_all_data, dmem_all_data->vector_gridk.d, U_array[0], num_rows);
         }
         dmem_all_data->iter.relax += 1;
      }
      
     // if (dmem_all_data->input.smoother == ASYNC_JACOBI || 
     //     dmem_all_data->input.smoother == ASYNC_L1_JACOBI){
     //    recv_gridk_flag = AsyncSmoothAddCorrect_LocalRes(dmem_all_data);
     //    vecop_begin = MPI_Wtime();
     //    DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, U_array[level], 1.0, num_rows);
     //    dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

     //    comm_begin = MPI_Wtime();
     //    SendRecv(dmem_all_data,
     //             &(dmem_all_data->comm.finestIntra_outsideSend),
     //             x_local_data,
     //             WRITE);
     //    SendRecv(dmem_all_data,
     //             &(dmem_all_data->comm.finestIntra_outsideRecv),
     //             x_ghost_data,
     //             READ);
     //    dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;

     //    residual_begin = MPI_Wtime();
     //    hypre_CSRMatrixMatvecOutOfPlace(-1.0,
     //                                    A_diag,
     //                                    hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x),
     //                                    1.0,
     //                                    hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.b),
     //                                    hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r),
     //                                    0);

     //    SendRecv(dmem_all_data,
     //             &(dmem_all_data->comm.finestIntra_outsideRecv),
     //             x_ghost_data,
     //             READ);
     //    DMEM_HypreParVector_Copy(Vtemp, dmem_all_data->vector_gridk.r, num_rows);
     //    hypre_CSRMatrixMatvecOutOfPlace(-1.0,
     //                                    A_offd,
     //                                    dmem_all_data->vector_gridk.x_ghost,
     //                                    1.0,
     //                                    hypre_ParVectorLocalVector(Vtemp),
     //                                    hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r),
     //                                    0);
     //    if (dmem_all_data->input.async_flag == 1){
     //       SendRecv(dmem_all_data,
     //                &(dmem_all_data->comm.finestIntra_outsideRecv),
     //                x_ghost_data,
     //                READ);
     //    }
     //    dmem_all_data->output.residual_wtime += MPI_Wtime() - residual_begin;
     // }
     // else if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
     //          dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL){
      if (dmem_all_data->input.smoother == ASYNC_JACOBI ||
          dmem_all_data->input.smoother == ASYNC_L1_JACOBI ||
          dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI){

         /* send to gridk neighbors */
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Set(dmem_all_data->vector_gridk.e, 0.0, num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
         if (update_flag == 1 || send_gridk_flag == 0){
            comm_begin = MPI_Wtime();
            send_gridk_flag = SendRecv(dmem_all_data,
                                       &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend),
                                       u_local_data,
                                       ACCUMULATE);
            dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
            if (update_flag == 1){
               vecop_begin = MPI_Wtime();
               DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.e, U_array[level], 1.0, num_rows);
               dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
            }
         }

         /* recv from gridk */
         comm_begin = MPI_Wtime();
         recv_gridk_flag = SendRecv(dmem_all_data,
                                    &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                                    e_local_data,
                                    ACCUMULATE);
         dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
         if ((dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
              dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL) &&
             (update_flag == 1 || recv_gridk_flag == 1)){
            vecop_begin = MPI_Wtime();
            DMEM_HypreParVector_Axpy(Vtemp, dmem_all_data->vector_gridk.e, 1.0, num_rows);
            if (dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL){
               DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.d, dmem_all_data->vector_gridk.e, 1.0, num_rows);
            }
            dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
         }

         /* if jacobi, send to intra now s.t. there is comm-comp overlap*/
         if (dmem_all_data->input.smoother == ASYNC_JACOBI ||
             dmem_all_data->input.smoother == ASYNC_L1_JACOBI){
            comm_begin = MPI_Wtime();
            send_intra_flag = SendRecv(dmem_all_data,
                                       &(dmem_all_data->comm.finestIntra_outsideSend),
                                       e_local_data,
                                       ACCUMULATE);
            dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
         }

         /* update diag residual */
         if (update_flag == 1 || recv_gridk_flag == 1){
            matvec_begin = residual_begin = MPI_Wtime();
            hypre_CSRMatrixMatvec(-1.0,
                                  A_diag,
                                  hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.e),
                                  1.0,
                                  hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r));
            matvec_end = MPI_Wtime();
            dmem_all_data->output.matvec_wtime += matvec_end - matvec_begin;
            dmem_all_data->output.residual_wtime += matvec_end - residual_begin;

            vecop_begin = MPI_Wtime();
            DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, dmem_all_data->vector_gridk.e, 1.0, num_rows);
            dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
         }

         /* recv from intra */
         vecop_begin = MPI_Wtime();
         DMEM_HypreRealArray_Set(x_ghost_data, 0.0, num_cols_offd);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
         comm_begin = MPI_Wtime();
         recv_intra_flag = SendRecv(dmem_all_data,
                                    &(dmem_all_data->comm.finestIntra_outsideRecv),
                                    x_ghost_data,
                                    ACCUMULATE);
         dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
 
         /* update offd residual */
         if (update_flag == 1 || recv_gridk_flag == 1 || recv_intra_flag == 1){
            if (recv_intra_flag == 1){
               matvec_begin = residual_begin = MPI_Wtime();
               hypre_CSRMatrixMatvec(-1.0,
                                     A_offd,
                                     dmem_all_data->vector_gridk.x_ghost,
                                     1.0,
                                     hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r));
               matvec_end = MPI_Wtime();
               dmem_all_data->output.residual_wtime += matvec_end - matvec_begin;
               dmem_all_data->output.residual_wtime += matvec_end - residual_begin;
            }

            if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
                dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL){
               vecop_begin = residual_norm_begin = MPI_Wtime();
               dmem_all_data->iter.r_L1norm_local = 0.0;
               for (int i = 0; i < num_rows; i++){
                  dmem_all_data->iter.r_L1norm_local += fabs(r_local_data[i]);
               }
               vecop_end = MPI_Wtime();
               dmem_all_data->output.residual_norm_wtime += vecop_end - residual_norm_begin;
               dmem_all_data->output.vecop_wtime += vecop_end - vecop_begin;
            }
         }
         
         if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
             dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL){ 
            /* send to intra */
            if (update_flag == 1 || send_intra_flag == 0){
               comm_begin = MPI_Wtime();
               send_intra_flag = SendRecv(dmem_all_data,
                                          &(dmem_all_data->comm.finestIntra_outsideSend),
                                          v_local_data,
                                          ACCUMULATE);
               dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
               vecop_begin = MPI_Wtime();
               DMEM_HypreParVector_Set(Vtemp, 0.0, num_rows);
               dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
            }

            double update_probability = StochasticParallelSouthwellUpdateProbability(dmem_all_data); 
            update_flag = 0;
            if (RandDouble(0.0, 1.0) < update_probability || dmem_all_data->comm.outside_recv_done_flag == 1){
               update_flag = 1;
            }
         }
         else {
            update_flag = 1;
         }
      }

      dmem_all_data->iter.cycle += 1;

      if (converge_flag == 1){
         break;
      }
   }
   
//   double end_begin = MPI_Wtime();
//   if (dmem_all_data->input.async_flag == 1){
//      DMEM_AsyncSmoothEnd(dmem_all_data);
//   }
//   //printf("%d %d\n", my_id, dmem_all_data->iter.relax);
//   dmem_all_data->output.end_wtime += MPI_Wtime() - end_begin;
//  // printf("%d %e\n", my_id, MPI_Wtime()-begin);
//  // hypre_ParVectorCopy(dmem_all_data->vector_gridk.r, F_array[level]);
//   dmem_all_data->comm.is_async_smoothing_flag = 0;
}

int AsyncSmoothCheckConverge(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->comm.async_smooth_done_flag == 1){
      if (DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), 2) == 1 &&
          DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend), 2) == 1){
         return 1;
      }
   }
   else {
      if (dmem_all_data->comm.outside_recv_done_flag == 1){
         if (DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend), 0) == 1 &&
             DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv), 0) == 1){
            dmem_all_data->comm.async_smooth_done_flag = 1;
         }
      }
      else {
         if (DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv), 0)){
            dmem_all_data->comm.outside_recv_done_flag = 1;
         }
      }
   }
   return 0;
}

int AsyncSmoothAddCorrect_LocalRes(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   double comm_begin, vecop_begin;
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
   
   comm_begin = MPI_Wtime();
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend),
            u_local_data,
            ACCUMULATE);
   recv_flag = SendRecv(dmem_all_data,
                        &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                        e_local_data,
                        ACCUMULATE);
   dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;

   vecop_begin = MPI_Wtime();
   if (recv_flag == 1){
      DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, dmem_all_data->vector_gridk.e, 1.0, num_rows);
   }
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
   return recv_flag;
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
      DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, dmem_all_data->vector_gridk.e, 1.0, num_rows);
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
   double comm_begin, vecop_begin;

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
      comm_begin = MPI_Wtime();
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
               e_local_data,
               ACCUMULATE);
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.finestIntra_outsideRecv),
               x_ghost_data,
               READ);      
      dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
      vecop_begin = MPI_Wtime();
      DMEM_HypreParVector_Axpy(dmem_all_data->vector_gridk.x, dmem_all_data->vector_gridk.e, 1.0, num_rows);
      dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
   }


   comm_begin = MPI_Wtime();
   CompleteInFlight(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend));
   dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;

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

void DMEM_AsyncSmoothEnd(DMEM_AllData *dmem_all_data)
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
   if (dmem_all_data->input.sps_probability_type != SPS_PROBABILITY_RANDOM){
      for (int i = 0; i < dmem_all_data->comm.finestIntra_outsideRecv.procs.size(); i++){
         if (dmem_all_data->iter.r_L1norm_local < dmem_all_data->comm.finestIntra_outsideRecv.r_norm[i]){
            x++;
         }
      }
   }
   
   double p;
   double alpha = dmem_all_data->input.sps_alpha;
   if (dmem_all_data->input.sps_probability_type == SPS_PROBABILITY_INVERSE){ 
      p = (1.0/x)*(1.0/alpha);
   }
   else if (dmem_all_data->input.sps_probability_type == SPS_PROBABILITY_EXPONENTIAL){
      p = exp(-x*alpha);
   }
   else {
      p = dmem_all_data->input.sps_alpha;
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
   double matvec_begin, vecop_begin;
   double matvec_end, vecop_end;

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

   vecop_begin = MPI_Wtime();
   DMEM_HypreParVector_Set(U_array[coarsest_level], 0.0, num_rows);
   if (dmem_all_data->input.smoother == L1_JACOBI){
      DMEM_HypreParVector_Ivaxpy(U_array[coarsest_level], F_array[coarsest_level], dmem_all_data->matrix.L1_row_norm_gridk[coarsest_level], num_rows);
   }
   else {
     DMEM_HypreParVector_Ivaxpy(U_array[coarsest_level], F_array[coarsest_level], dmem_all_data->matrix.wJacobi_scale_gridk[coarsest_level], num_rows);
   }
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

   if (dmem_all_data->input.simple_jacobi_flag == 0){
      matvec_begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvec(1.0,
                               A_array[coarsest_level],
                               U_array[coarsest_level],
                               0.0,
                               Vtemp);
      dmem_all_data->output.matvec_wtime += MPI_Wtime() - matvec_begin;

      vecop_begin = MPI_Wtime();
      DMEM_HypreParVector_Scale(U_array[coarsest_level], 2.0, num_rows);
      if (dmem_all_data->input.smoother == L1_JACOBI){
         DMEM_HypreParVector_Ivaxpy(U_array[coarsest_level], Vtemp, dmem_all_data->matrix.symmL1_row_norm_gridk[coarsest_level], num_rows);
      }
      else {
         DMEM_HypreParVector_Ivaxpy(U_array[coarsest_level], Vtemp, dmem_all_data->matrix.symmwJacobi_scale_gridk[coarsest_level], num_rows);
      }
      dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
   }
}
