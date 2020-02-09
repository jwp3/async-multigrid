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
           // begin = MPI_Wtime();
            hypre_MPI_Wait(&(comm_data->requests_inflight[i][j]), MPI_STATUS_IGNORE);
           // dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - begin;
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
   double vecop_begin, comp_begin, mpiisend_begin, mpiirecv_begin;
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
         vecop_begin = MPI_Wtime();
         if (comm_data->type == FINE_INTRA_INSIDE_SEND){
            for (int j = 0; j < vec_len; j++){
               comm_data->data[i][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
            }
         }
         else {
            DMEM_HypreRealArray_Copy(comm_data->data[i], &v[vec_start], vec_len);
         }
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

         mpiisend_begin = MPI_Wtime();
         hypre_MPI_Isend(comm_data->data[i],
                         vec_len+1,
                         HYPRE_MPI_REAL,
                         ip,
                         comm_data->tag,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
         dmem_all_data->output.mpiisend_wtime += MPI_Wtime() - mpiisend_begin;
         comm_data->message_count[i]++;
         return_flag = 1;
      }
      /* inside recv */
      else if (comm_data->type == GRIDK_INSIDE_RECV || comm_data->type == FINE_INTRA_INSIDE_RECV){
         mpiirecv_begin = MPI_Wtime();
         hypre_MPI_Irecv(comm_data->data[i],
                         vec_len+1,
                         HYPRE_MPI_REAL,
                         ip,
                         comm_data->tag,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
         dmem_all_data->output.mpiirecv_wtime += MPI_Wtime() - mpiirecv_begin;
         comm_data->message_count[i]++;
         comm_data->recv_flags[i] = 1;
         return_flag = 1;
      }
      /* outside send */
      else if (comm_data->type == GRIDK_OUTSIDE_SEND || comm_data->type == FINE_INTRA_OUTSIDE_SEND){
         /* async outside send */
         if (dmem_all_data->input.async_flag == 1){
            if (comm_data->done_flags[i] < 2){
               vecop_begin = MPI_Wtime();
               if (op == WRITE){
                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
                     for (int j = 0; j < vec_len; j++){
                        comm_data->data[i][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
                     }
                  }
                  else {
                     DMEM_HypreRealArray_Copy(comm_data->data[i], &v[vec_start], vec_len);
                  }
               }
               else {
                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
                     for (int j = 0; j < vec_len; j++){
                        comm_data->data[i][j] += v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
                     }
                  }
                  else {
                     DMEM_HypreRealArray_Axpy(comm_data->data[i], &v[vec_start], 1.0, vec_len);
                  }
               }
               dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
              
               CheckInFlight(dmem_all_data, comm_data, i);
               if (comm_data->num_inflight[i] < comm_data->max_inflight[i]){
                  int next_inflight = comm_data->next_inflight[i];

                  vecop_begin = MPI_Wtime();
                  DMEM_HypreRealArray_Copy(comm_data->data_inflight[i][next_inflight], comm_data->data[i], vec_len);
                  DMEM_HypreRealArray_Set(comm_data->data[i], 0.0, vec_len);
                  dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

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
                     if (dmem_all_data->iter.cycle >= num_cycles-2 ||
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
                  if ((dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
                       dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL) &&
                      comm_data->type == FINE_INTRA_OUTSIDE_SEND){
                     comm_data->data_inflight[i][next_inflight][vec_len+1] = dmem_all_data->iter.r_L1norm_local;
                  }

                  mpiisend_begin = MPI_Wtime();
                  hypre_MPI_Isend(comm_data->data_inflight[i][next_inflight],
                                  send_len,
                                  HYPRE_MPI_REAL,
                                  ip,
                                  comm_data->tag,
                                  MPI_COMM_WORLD,
                                  &(comm_data->requests_inflight[i][next_inflight]));
                  dmem_all_data->output.mpiisend_wtime += MPI_Wtime() - mpiisend_begin;
                  comm_data->inflight_flags[i][next_inflight] = 1;
                  comm_data->num_inflight[i]++;
                  SetNextInFlight(comm_data, i);
                  comm_data->message_count[i]++;
                  dmem_all_data->output.num_messages++;
                  return_flag = 1;
               }
            }
         }
         else { /* sync outside send */
            vecop_begin = MPI_Wtime();
            if (comm_data->type == FINE_INTRA_INSIDE_SEND){
               for (int j = 0; j < vec_len; j++){
                  comm_data->data[i][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
               }
            }
            else {
               DMEM_HypreRealArray_Copy(comm_data->data[i], &v[vec_start], vec_len);
            }
            dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;          

            mpiisend_begin = MPI_Wtime();
            hypre_MPI_Isend(comm_data->data[i],
                            vec_len+1,
                            HYPRE_MPI_REAL,
                            ip,
                            comm_data->tag,
                            MPI_COMM_WORLD,
                            &(comm_data->requests[i]));
            dmem_all_data->output.mpiisend_wtime += MPI_Wtime() - mpiisend_begin;
            comm_data->message_count[i]++;
           // dmem_all_data->output.num_messages++;
            return_flag = 1;
         }
      }
      /* outisde recv */
      else if (comm_data->type == GRIDK_OUTSIDE_RECV || comm_data->type == FINE_INTRA_OUTSIDE_RECV){
         /* async outisde recv */
         if (dmem_all_data->input.async_flag == 1){
            if (comm_data->done_flags[i] < 2){
               while (1) {
                  flag = 0;
                  hypre_MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
                  if (flag){
                     vecop_begin = MPI_Wtime();
                     if (op == ACCUMULATE){
                        DMEM_HypreRealArray_Axpy(&v[vec_start], comm_data->data[i], 1.0, vec_len);
                     }
                     else if (op == READ){
                        DMEM_HypreRealArray_Copy(&v[vec_start], comm_data->data[i], vec_len);
                     }
                     dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

                     int recv_len = vec_len+2;
                     if ((dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
                          dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL) &&
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
                     mpiirecv_begin = MPI_Wtime();
                     hypre_MPI_Irecv(comm_data->data[i],
                                     recv_len,
                                     HYPRE_MPI_REAL,
                                     ip,
                                     comm_data->tag,
                                     MPI_COMM_WORLD,
                                     &(comm_data->requests[i]));
                     dmem_all_data->output.mpiirecv_wtime += MPI_Wtime() - mpiirecv_begin;
                     comm_data->recv_flags[i] = 1;
                     return_flag = 1;
                  }
                  else {
                     break;
                  }
               }
            }
         }
         else { /* sync outisde recv */
            comm_data->message_count[i]++;
            mpiirecv_begin = MPI_Wtime();
            hypre_MPI_Irecv(comm_data->data[i],
                            vec_len+1,
                            HYPRE_MPI_REAL,
                            ip,
                            comm_data->tag,
                            MPI_COMM_WORLD,
                            &(comm_data->requests[i]));
            dmem_all_data->output.mpiirecv_wtime += MPI_Wtime() - mpiirecv_begin;
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
   double vecop_begin, mpiwait_begin;

   mpiwait_begin = MPI_Wtime();
   hypre_MPI_Waitall(comm_data->procs.size(),
                     comm_data->requests,
                     hypre_MPI_STATUSES_IGNORE);
   dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - mpiwait_begin;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);

    
   vecop_begin = MPI_Wtime();
   for (int i = 0; i < comm_data->procs.size(); i++){
      HYPRE_Int vec_start = comm_data->start[i];
      HYPRE_Int vec_len = comm_data->len[i];
      if (op == ACCUMULATE){
         DMEM_HypreRealArray_Axpy(&v[vec_start], comm_data->data[i], 1.0, vec_len);
      }
      else if (op == READ){
         DMEM_HypreRealArray_Copy(&v[vec_start], comm_data->data[i], vec_len);
      }
   }
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
}
