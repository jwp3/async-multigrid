#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Add.hpp"
#include "Misc.hpp"
#include "seq_mv.h"
#include "_hypre_utilities.h"
#include "DMEM_Misc.hpp"

using namespace std;

void CompleteInFlight(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   double begin;
   for (int i = 0; i < comm_data->procs.size(); i++){
      for (int j = 0; j < comm_data->max_inflight[i]; j++){
         if (comm_data->inflight_flags[i][j] == 1){
            begin = MPI_Wtime();
            hypre_MPI_Wait(&(comm_data->requests_inflight[i][j]), MPI_STATUS_IGNORE);
            dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - begin;
         }
      }
   }
}

void CheckInFlight(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int i)
{
   int flag;
   double begin;
   while (1){
      int break_flag = 0;
      if (dmem_all_data->comm.all_done_flag == 0){
         break_flag = 1;
      }
      else {
         if (comm_data->num_inflight[i] < comm_data->max_inflight[i]){
            break;
         }
      }
      for (int j = 0; j < comm_data->max_inflight[i]; j++){ 
         if (comm_data->inflight_flags[i][j] == 1){
            begin = MPI_Wtime();
            hypre_MPI_Test(&(comm_data->requests_inflight[i][j]), &flag, MPI_STATUS_IGNORE);
            dmem_all_data->output.mpitest_wtime += MPI_Wtime() - begin;
            if (flag){
               comm_data->inflight_flags[i][j] = 0;
               comm_data->num_inflight[i]--;
               if (j < comm_data->next_inflight[i]){
                  comm_data->next_inflight[i] = j;
               }
               if (dmem_all_data->comm.all_done_flag == 1){
                  break_flag = 1;
                  break;
               }
            }
         }
         else {
            if (dmem_all_data->comm.all_done_flag == 1){
               break_flag = 1;
               break;
            }
         }
      }
      if (break_flag == 1){
         break;
      }
   }
}

void SetNextInFlight(DMEM_CommData *comm_data, int i)
{
   int flag;
   for (int j = 0; j < comm_data->max_inflight[i]; j++){
      if (comm_data->inflight_flags[i][j] == 0){
         comm_data->next_inflight[i] = j;
         return;
      }
   }
   comm_data->next_inflight[i] = comm_data->max_inflight[i];
}

int SendRecv(DMEM_AllData *dmem_all_data,
             DMEM_CommData *comm_data,
             HYPRE_Real *v,
             HYPRE_Int op)
{
   HYPRE_Int flag;
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   int my_grid = dmem_all_data->grid.my_grid;
   double comp_begin, begin;
   int return_flag = 0;

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(dmem_all_data->matrix.A_gridk);


   for (int i = 0; i < comm_data->procs.size(); i++){
      flag = 0;
      comm_data->recv_flags[i] = 0;
      HYPRE_Int ip = comm_data->procs[i];
      HYPRE_Int vec_start = comm_data->start[i];
      HYPRE_Int vec_end = comm_data->end[i];
      HYPRE_Int vec_len = comm_data->len[i];
      /* inside send */
      if (comm_data->type == GRIDK_INSIDE_SEND || comm_data->type == FINE_INTRA_INSIDE_SEND){
//#if defined(HYPRE_USING_CUDA)
//         if (comm_data->type == FINE_INTRA_INSIDE_SEND){
//           // PackOnDevice(comm_data->data[i], v, hypre_ParCSRCommPkgSendMapElmts(comm_pkg), vec_start, vec_end, HYPRE_STREAM(4));
//         }
//         else {
//            DMEM_HypreRealArray_Copy(comm_data->data[i], &v[vec_start], vec_len);
//         }
//#else
         if (comm_data->type == FINE_INTRA_INSIDE_SEND){
            for (int j = 0; j < vec_len; j++){
               comm_data->data[i][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
            }
         }
         else {
            for (int j = 0; j < vec_len; j++){
               comm_data->data[i][j] = v[vec_start+j];
            }
         }
//#endif
         begin = MPI_Wtime();
         hypre_MPI_Isend(comm_data->data[i],
                         vec_len+1,
                         HYPRE_MPI_REAL,
                         ip,
                         comm_data->tag,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
         dmem_all_data->output.mpiisend_wtime += MPI_Wtime() - begin;
         comm_data->message_count[i]++;
         return_flag = 1;
      }
      /* inside recv */
      else if (comm_data->type == GRIDK_INSIDE_RECV || comm_data->type == FINE_INTRA_INSIDE_RECV){
         begin = MPI_Wtime();
//#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
//         SetAsyncMode(1);
//         HypreComplexPrefetchToDevice(comm_data->data[i], vec_len);
//         cudaStreamSynchronize(HYPRE_STREAM(4));
//         SetAsyncMode(0);
//#endif
         hypre_MPI_Irecv(comm_data->data[i],
                         vec_len+1,
                         HYPRE_MPI_REAL,
                         ip,
                         comm_data->tag,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
         dmem_all_data->output.mpiirecv_wtime += MPI_Wtime() - begin;
         comm_data->message_count[i]++;
         comm_data->recv_flags[i] = 1;
         return_flag = 1;
      }
      /* outside send */
      else if (comm_data->type == GRIDK_OUTSIDE_SEND || comm_data->type == FINE_INTRA_OUTSIDE_SEND){
         if (dmem_all_data->input.async_flag == 1){ /* async outside send */
            if (comm_data->done_flags[i] < 2){
//#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
//               if (op == WRITE){
//                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
//                     PackOnDevice(comm_data->data[i], v, hypre_ParCSRCommPkgSendMapElmts(comm_pkg), vec_start, vec_end, HYPRE_STREAM(4));
//                  }
//                  else {
//                     HypreComplexPrefetchToDevice(comm_data->data[i], vec_len);
//                     HypreComplexPrefetchToDevice(&v[vec_start], vec_len);
//                     VecCopy(comm_data->data[i], &v[vec_start], vec_len, HYPRE_STREAM(4));
//                     cudaStreamSynchronize(HYPRE_STREAM(4));
//                  }
//               }
//               else {
//                  HYPRE_Real alpha = 1.0;
//                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
//                    // PackOnDeviceAxpy(comm_data->data[i], v, hypre_ParCSRCommPkgSendMapElmts(comm_pkg), alpha, vec_start, vec_end, HYPRE_STREAM(4));
//                  }
//                  else {
//                     HypreComplexPrefetchToDevice(comm_data->data[i], vec_len);
//                     HypreComplexPrefetchToDevice(&v[vec_start], vec_len);
//                     static cublasHandle_t handle;
//                     static HYPRE_Int firstcall = 1;
//                     if (firstcall){
//                        handle = getCublasHandle();
//                        firstcall = 0;
//                     }
//                     cublasDaxpy(handle, vec_len, &alpha, &v[vec_start], 1, comm_data->data[i], 1);
//                     cudaStreamSynchronize(HYPRE_STREAM(4));
//                  }
//               }
//#else

//#if defined(HYPRE_USING_CUDA)
//               if (op == WRITE){
//                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
//                     // DMEM_HypreRealArray_Prefetch(&v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start)], , HYPRE_MEMORY_DEVICE);
//                      HYPRE_THRUST_CALL(gather,
//                                        hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
//                                        hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1),
//                                        &v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start)],
//                                        comm_data->data[i]);
//                  }
//                  else {
//                     DMEM_HypreRealArray_Copy(comm_data->data[i], &v[vec_start], vec_len);
//                  }
//               }
//               else {
//                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
//                     // PackOnDeviceAxpy(comm_data->data[i], v, hypre_ParCSRCommPkgSendMapElmts(comm_pkg), alpha, vec_start, vec_end, HYPRE_STREAM(4));
//                  }
//                  else {
//                     DMEM_HypreRealArray_Axpy(comm_data->data[i], &v[vec_start], 1.0, vec_len); 
//                  }
//               }
//#else
               if (op == WRITE){
                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
                     for (int j = 0; j < vec_len; j++){
                        comm_data->data[i][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
                     }
                  }
                  else {
                     for (int j = 0; j < vec_len; j++){
                        comm_data->data[i][j] = v[vec_start+j];
                     }
                  }
               }
               else {
                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
                     for (int j = 0; j < vec_len; j++){
                        comm_data->data[i][j] += v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
                     }
                  }
                  else {
                     for (int j = 0; j < vec_len; j++){
                        comm_data->data[i][j] += v[vec_start+j];
                     }
                  }
               }
//#endif
              
               CheckInFlight(dmem_all_data, comm_data, i);
               if (comm_data->num_inflight[i] < comm_data->max_inflight[i]){
                  int next_inflight = comm_data->next_inflight[i];
//#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
//                  HypreComplexPrefetchToDevice(comm_data->data[i], vec_len);
//                  HypreComplexPrefetchToDevice(comm_data->data_inflight[i][next_inflight], vec_len);
//                  VecCopy(comm_data->data_inflight[i][next_inflight], comm_data->data[i], vec_len, HYPRE_STREAM(4));
//                  VecSet(comm_data->data[i], vec_len, 0.0, HYPRE_STREAM(4));
//                  cudaStreamSynchronize(HYPRE_STREAM(4));
//#else

//#if defined(HYPRE_USING_CUDA)
//                  DMEM_HypreRealArray_Copy(comm_data->data_inflight[i][next_inflight], comm_data->data[i], vec_len);
//                  DMEM_HypreRealArray_Set(comm_data->data[i], 0.0, vec_len);
//#else
                  for (int j = 0; j < vec_len; j++){
                     comm_data->data_inflight[i][next_inflight][j] = comm_data->data[i][j];
                     comm_data->data[i][j] = 0.0;
                  }
//#endif
                  int num_cycles;
                  if (dmem_all_data->input.solver == MULT_MULTADD){
                     num_cycles = dmem_all_data->input.num_inner_cycles;
                  }
                  else {
                     num_cycles = dmem_all_data->input.num_cycles;
                  }

                  int my_converge_flag = 0;
                  int all_done_flag;
                  if (dmem_all_data->comm.is_async_smoothing_flag == 1){
                     all_done_flag = dmem_all_data->comm.async_smooth_done_flag;
                     if (dmem_all_data->comm.outside_recv_done_flag == 1){
                        my_converge_flag = 1;
                     }
                  }
                  else {
                     all_done_flag = dmem_all_data->comm.all_done_flag;
                     if (dmem_all_data->iter.cycle >= num_cycles-1 ||
                         dmem_all_data->iter.r_L2norm_local_converge_flag == 1){
                        my_converge_flag = 1;
                     }
                  }
                  if (my_converge_flag == 1){
                     comm_data->data_inflight[i][next_inflight][vec_len] = 1.0;
                     if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
                        comm_data->done_flags[i] = 2;
                     }
                     else {
                        comm_data->done_flags[i] = 1;
                        if (all_done_flag == 1){
                           comm_data->done_flags[i] = 2;
                           comm_data->data_inflight[i][next_inflight][vec_len] = 2.0;
                        }
                     }
                  }

                  int send_len = vec_len+2;
                  if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL &&
                      comm_data->type == FINE_INTRA_OUTSIDE_SEND){
                     comm_data->data_inflight[i][next_inflight][vec_len+1] = dmem_all_data->iter.r_L1norm_local;
                  }

                  begin = MPI_Wtime();
                  hypre_MPI_Isend(comm_data->data_inflight[i][next_inflight],
                                  send_len,
                                  HYPRE_MPI_REAL,
                                  ip,
                                  comm_data->tag,
                                  MPI_COMM_WORLD,
                                  &(comm_data->requests_inflight[i][next_inflight]));
                  dmem_all_data->output.mpiisend_wtime += MPI_Wtime() - begin;
                  comm_data->inflight_flags[i][next_inflight] = 1;
                  comm_data->num_inflight[i]++;
                  SetNextInFlight(comm_data, i);
                  comm_data->message_count[i]++;
                  return_flag = 1;
               }
            }
         }
         else { /* sync outside send */
//#if defined(HYPRE_USING_CUDA)
//            if (comm_data->type == FINE_INTRA_INSIDE_SEND){
//              // PackOnDevice(comm_data->data[i], v, hypre_ParCSRCommPkgSendMapElmts(comm_pkg), vec_start, vec_end, HYPRE_STREAM(4));
//            }
//            else {
//               DMEM_HypreRealArray_Copy(comm_data->data[i], &v[vec_start], vec_len);
//            }
//#else
            if (comm_data->type == FINE_INTRA_INSIDE_SEND){
               for (int j = 0; j < vec_len; j++){
                  comm_data->data[i][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
               }
            }
            else {
               for (int j = 0; j < vec_len; j++){
                  comm_data->data[i][j] = v[vec_start+j];
               }
            }
//#endif
            begin = MPI_Wtime();
            hypre_MPI_Isend(comm_data->data[i],
                            vec_len+1,
                            HYPRE_MPI_REAL,
                            ip,
                            comm_data->tag,
                            MPI_COMM_WORLD,
                            &(comm_data->requests[i]));
            dmem_all_data->output.mpiisend_wtime += MPI_Wtime() - begin;
            comm_data->message_count[i]++;
            return_flag = 1;
         }
      }
      /* outisde recv */
      else if (comm_data->type == GRIDK_OUTSIDE_RECV || comm_data->type == FINE_INTRA_OUTSIDE_RECV){
         if (dmem_all_data->input.async_flag == 1){
            if (comm_data->done_flags[i] < 2){
               while (1) {
                  flag = 0;
                  hypre_MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
                  if (flag){
//#if defined(HYPRE_USING_CUDA)
//                   //  HypreComplexPrefetchToDevice(comm_data->data[i], vec_len);
//                   //  HypreComplexPrefetchToDevice(&v[vec_start], vec_len);
//                     if (op == ACCUMULATE){
//                       // HYPRE_Real alpha = 1.0;
//                       // static cublasHandle_t handle;
//                       // static HYPRE_Int firstcall = 1;
//                       // if (firstcall){
//                       //    handle = getCublasHandle();
//                       //    firstcall = 0;
//                       // }
//                       // cublasDaxpy(handle, vec_len, &alpha, comm_data->data[i], 1, &v[vec_start], 1);
//                        DMEM_HypreRealArray_Axpy(&v[vec_start], comm_data->data[i], 1.0, vec_len);
//                     }
//                     else if (op == READ){
//                        DMEM_HypreRealArray_Copy(&v[vec_start], comm_data->data[i], vec_len);
//                     }
//                    // cudaStreamSynchronize(HYPRE_STREAM(4));
//#else
                     if (op == ACCUMULATE){
                        for (int j = 0; j < vec_len; j++){
                           v[vec_start+j] += comm_data->data[i][j];
                        }
                     }
                     else if (op == READ){
                        for (int j = 0; j < vec_len; j++){
                           v[vec_start+j] = comm_data->data[i][j];
                        }
                     }
//#endif
                     if (comm_data->update_res_in_comm == 1){
                        comp_begin = MPI_Wtime();
                        HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(dmem_all_data->matrix.A_gridk);
                        HYPRE_Real *x_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost);
                        HYPRE_Real *r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r));
                        if (dmem_all_data->input.res_update_type == RES_ACCUMULATE){
                           for (int j = 0; j < num_rows; j++){
                              for (int k = 0; k < comm_data->a_ghost_j[i][j].size(); k++){
                                 int ii = comm_data->a_ghost_j[i][j][k];
                                 HYPRE_Real aij = comm_data->a_ghost_data[i][j][k];
                                 r_local_data[j] -= aij * x_ghost_data[ii];
                              }
                           }
                        }
                        else {
                           HYPRE_Real *x_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost);
                           HYPRE_Real *x_ghost_prev_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost_prev);
                           for (int j = 0; j < num_rows; j++){
                              for (int k = 0; k < comm_data->a_ghost_j[i][j].size(); k++){
                                 int ii = comm_data->a_ghost_j[i][j][k];
                                 HYPRE_Real aij = comm_data->a_ghost_data[i][j][k];
                                 r_local_data[j] += aij * x_ghost_data[ii] - aij * x_ghost_prev_data[ii];
                                 x_ghost_prev_data[ii] = x_ghost_data[ii];
                              }
                           }
                        }
                        dmem_all_data->output.comp_wtime += MPI_Wtime() - comp_begin;
                     }

                     int recv_len = vec_len+2;
                     if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL &&
                         comm_data->type == FINE_INTRA_OUTSIDE_RECV){
                        comm_data->r_norm[i] = comm_data->data[i][vec_len+1];
                     }

                     if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
                        if (comm_data->data[i][vec_len] == 1.0){
                           comm_data->done_flags[i] = 2;
                           break;
                        }
                     }
                     else {
                        if (comm_data->data[i][vec_len] == 1.0){
                           comm_data->done_flags[i] = 1;
                        }
                        else if (comm_data->data[i][vec_len] == 2.0){
                           comm_data->done_flags[i] = 2;
                           break;
                        }
                        else {
                        }
                     }
                     comm_data->message_count[i]++;
                     begin = MPI_Wtime();
                     hypre_MPI_Irecv(comm_data->data[i],
                                     recv_len,
                                     HYPRE_MPI_REAL,
                                     ip,
                                     comm_data->tag,
                                     MPI_COMM_WORLD,
                                     &(comm_data->requests[i]));
                     dmem_all_data->output.mpiirecv_wtime += MPI_Wtime() - begin;
                     comm_data->recv_flags[i] = 1;
                     return_flag = 1;
                  }
                  else {
                     break;
                  }
               }
            }
         }
         else {
//#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
//            SetAsyncMode(1);
//            HypreComplexPrefetchToDevice(comm_data->data[i], vec_len);
//            cudaStreamSynchronize(HYPRE_STREAM(4));
//            SetAsyncMode(0);
//#endif
            comm_data->message_count[i]++;
            begin = MPI_Wtime();
            hypre_MPI_Irecv(comm_data->data[i],
                            vec_len+1,
                            HYPRE_MPI_REAL,
                            ip,
                            comm_data->tag,
                            MPI_COMM_WORLD,
                            &(comm_data->requests[i]));
            dmem_all_data->output.mpiirecv_wtime += MPI_Wtime() - begin;
            comm_data->recv_flags[i] = 1;
            return_flag = 1;
         }
      }
   }
   return return_flag;
}

void CompleteRecv(DMEM_AllData *dmem_all_data,
                  DMEM_CommData *comm_data,
                  HYPRE_Real *v,
                  HYPRE_Int op)
{
   double begin;

   begin = MPI_Wtime();
   hypre_MPI_Waitall(comm_data->procs.size(),
                     comm_data->requests,
                     hypre_MPI_STATUSES_IGNORE);
   dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - begin;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
 
   for (int i = 0; i < comm_data->procs.size(); i++){
      HYPRE_Int vec_start = comm_data->start[i];
      HYPRE_Int vec_len = comm_data->len[i];
//#if defined(HYPRE_USING_CUDA)
//    //  HypreComplexPrefetchToDevice(comm_data->data[i], vec_len);
//    //  HypreComplexPrefetchToDevice(&v[vec_start], vec_len);
//      if (op == ACCUMULATE){
//        // HYPRE_Real alpha = 1.0;
//        // static cublasHandle_t handle;
//        // static HYPRE_Int firstcall = 1;
//        // if (firstcall){
//        //    handle = getCublasHandle();
//        //    firstcall = 0;
//        // }
//        // cublasDaxpy(handle, vec_len, &alpha, comm_data->data[i], 1, &v[vec_start], 1);
//         DMEM_HypreRealArray_Axpy(&v[vec_start], comm_data->data[i], 1.0, vec_len);
//      }
//      else if (op == READ){
//         DMEM_HypreRealArray_Copy(&v[vec_start], comm_data->data[i], vec_len);
//      }
//     // cudaStreamSynchronize(HYPRE_STREAM(4));
//#else
      if (op == ACCUMULATE){
         for (HYPRE_Int j = 0; j < vec_len; j++){
            v[vec_start+j] += comm_data->data[i][j];
         }
      }
      else if (op == READ){
         for (HYPRE_Int j = 0; j < vec_len; j++){
            v[vec_start+j] = comm_data->data[i][j];
         }
      }
//#endif
   }
}
