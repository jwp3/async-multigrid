#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Add.hpp"
#include "Misc.hpp"

using namespace std;

void CompleteInFlight(DMEM_CommData *comm_data)
{
   for (int i = 0; i < comm_data->procs.size(); i++){
      for (int j = 0; j < comm_data->max_inflight[i]; j++){
         hypre_MPI_Wait(&(comm_data->requests_inflight[i][j]), MPI_STATUS_IGNORE);
      }
   }
}

void CheckInFlight(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int i)
{
   int flag;
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
            hypre_MPI_Test(&(comm_data->requests_inflight[i][j]), &flag, MPI_STATUS_IGNORE);
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

void SendRecv(DMEM_AllData *dmem_all_data,
              DMEM_CommData *comm_data,
              HYPRE_Real *v,
              HYPRE_Int op)
{
   HYPRE_Int flag;
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(dmem_all_data->matrix.A_fine);

   for (int i = 0; i < comm_data->procs.size(); i++){
      flag = 0;
      HYPRE_Int ip = comm_data->procs[i];
      HYPRE_Int vec_start = comm_data->start[i];
      HYPRE_Int vec_len = comm_data->len[i];
      /* inside send */
      if (comm_data->type == GRIDK_INSIDE_SEND || comm_data->type == FINE_INTRA_INSIDE_SEND){
         if (comm_data->type == FINE_INTRA_INSIDE_SEND){
            for (HYPRE_Int j = 0; j < vec_len; j++){
               comm_data->data[i][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
            }
         }
         else {
            for (HYPRE_Int j = 0; j < vec_len; j++){
               comm_data->data[i][j] = v[vec_start+j];
            }
         }
         hypre_MPI_Isend(comm_data->data[i],
                         vec_len+1,
                         HYPRE_MPI_REAL,
                         ip,
                         comm_data->tag,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
         comm_data->message_count[i]++;
      }
      /* inside recv */
      else if (comm_data->type == GRIDK_INSIDE_RECV || comm_data->type == FINE_INTRA_INSIDE_RECV){
         hypre_MPI_Irecv(comm_data->data[i],
                         vec_len+1,
                         HYPRE_MPI_REAL,
                         ip,
                         comm_data->tag,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
         comm_data->message_count[i]++;
      }
      /* outside send */
      else if (comm_data->type == GRIDK_OUTSIDE_SEND || comm_data->type == FINE_INTRA_OUTSIDE_SEND){
         if (dmem_all_data->input.async_flag == 1){
            if (comm_data->done_flags[i] < 2){
               CheckInFlight(dmem_all_data, comm_data, i);
               if (comm_data->num_inflight[i] < comm_data->max_inflight[i]){
                  int next_inflight = comm_data->next_inflight[i];
                  if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
                     for (HYPRE_Int j = 0; j < vec_len; j++){
                        comm_data->data_inflight[i][next_inflight][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
                     }
                  }
                  else {
                     for (HYPRE_Int j = 0; j < vec_len; j++){
                        comm_data->data_inflight[i][next_inflight][j] = v[vec_start+j];
                     }
                  }
                  int num_cycles;
                  if (dmem_all_data->input.solver == MULT_MULTADD){
                     num_cycles = dmem_all_data->input.num_inner_cycles;
                  }
                  else {
                     num_cycles = dmem_all_data->input.num_cycles;
                  }
                  if (dmem_all_data->iter.cycle >= num_cycles-1 ||
                      dmem_all_data->iter.r_norm2_local_converge_flag == 1){
                     comm_data->data_inflight[i][next_inflight][vec_len] = 1.0;
                     if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
                        comm_data->done_flags[i] = 2;
                     }
                     else {
                        comm_data->done_flags[i] = 1;
                        if (dmem_all_data->comm.all_done_flag == 1){
                           comm_data->done_flags[i] = 2;
                           comm_data->data_inflight[i][next_inflight][vec_len] = 2.0;
                        }
                     }
                  }
                  hypre_MPI_Isend(comm_data->data_inflight[i][next_inflight],
                                  vec_len+1,
                                  HYPRE_MPI_REAL,
                                  ip,
                                  comm_data->tag,
                                  MPI_COMM_WORLD,
                                  &(comm_data->requests_inflight[i][next_inflight]));
                  comm_data->inflight_flags[i][next_inflight] = 1;
                  comm_data->num_inflight[i]++;
                  SetNextInFlight(comm_data, i);
                  comm_data->message_count[i]++;
               }
            }
         }
         else {
            if (comm_data->type == FINE_INTRA_OUTSIDE_SEND){
               for (HYPRE_Int j = 0; j < vec_len; j++){
                  comm_data->data[i][j] = v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, vec_start+j)];
               }
            }
            else {
               for (HYPRE_Int j = 0; j < vec_len; j++){
                  comm_data->data[i][j] = v[vec_start+j];
               }
            }
            hypre_MPI_Isend(comm_data->data[i],
                            vec_len+1,
                            HYPRE_MPI_REAL,
                            ip,
                            comm_data->tag,
                            MPI_COMM_WORLD,
                            &(comm_data->requests[i]));
            comm_data->message_count[i]++;
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
                     hypre_MPI_Irecv(comm_data->data[i],
                                     vec_len+1,
                                     HYPRE_MPI_REAL,
                                     ip,
                                     comm_data->tag,
                                     MPI_COMM_WORLD,
                                     &(comm_data->requests[i]));
                  }
                  else {
                     break;
                  }
               };
            }
         }
         else {
            comm_data->message_count[i]++;
            hypre_MPI_Irecv(comm_data->data[i],
                            vec_len+1,
                            HYPRE_MPI_REAL,
                            ip,
                            comm_data->tag,
                            MPI_COMM_WORLD,
                            &(comm_data->requests[i]));
         }
      }
   }
}

void CompleteRecv(DMEM_AllData *dmem_all_data,
                  DMEM_CommData *comm_data,
                  HYPRE_Real *v,
                  HYPRE_Int op)
{
   hypre_MPI_Waitall(comm_data->procs.size(),
                     comm_data->requests,
                     hypre_MPI_STATUSES_IGNORE);
 
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      HYPRE_Int vec_start = comm_data->start[i];
      HYPRE_Int vec_len = comm_data->len[i];
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
   }   
}
