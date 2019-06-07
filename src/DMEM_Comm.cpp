#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Add.hpp"
#include "Misc.hpp"

using namespace std;

//void FineIntraSendRecv(DMEM_AllData *dmem_all_data,
//                       DMEM_CommData *comm_data,
//                       HYPRE_Real *v,
//                       HYPRE_Int op)
//{
//   int my_id;
//   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
//   MPI_Status status;
//   HYPRE_Int flag;
//   hypre_ParCSRMatrix *A = dmem_all_data->matrix.A_fine;
//   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
//   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
//      flag = 0;
//      HYPRE_Int ip = comm_data->procs[i];
//      HYPRE_Int vec_start = comm_data->start[i];
//      HYPRE_Int vec_end = comm_data->end[i];
//      HYPRE_Int vec_len = comm_data->len[i];
//      switch(comm_data->type){
//         /* fine inside send */
//         case FINE_INTRA_INSIDE_SEND:
//            for (HYPRE_Int j = vec_start; j < vec_end; j++){
//               dmem_all_data->comm.fine_send_data[j] = 
//                  v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
//            }
//            hypre_MPI_Isend(&(dmem_all_data->comm.fine_send_data[vec_start]),
//                            vec_len,
//                            HYPRE_MPI_REAL,
//                            ip,
//                            comm_data->tag,
//                            MPI_COMM_WORLD,
//                            &(comm_data->requests[i]));
//            comm_data->message_count[i]++;
//            break;
//         /* fine inside recv */
//         case FINE_INTRA_INSIDE_RECV:
//            hypre_MPI_Irecv(&(dmem_all_data->comm.fine_recv_data[vec_start]),
//                            vec_len,
//                            HYPRE_MPI_REAL,
//                            ip,
//                            comm_data->tag,
//                            MPI_COMM_WORLD,
//                            &(comm_data->requests[i]));
//            comm_data->message_count[i]++;
//            break;
//         /* fine outside send */
//         case FINE_INTRA_OUTSIDE_SEND:
//            if (dmem_all_data->input.async_flag == 1){
//               if (comm_data->done_flags[i] < 2){
//                  hypre_MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
//                  if (flag){
//                     if (comm_data->message_count[i] >= dmem_all_data->input.num_cycles-1 ||
//                         dmem_all_data->iter.r_norm2_local_converge_flag == 1){
//                       // comm_data->data[i][vec_len] = 1.0;
//                        if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
//                           comm_data->done_flags[i] = 2;
//                        }
//                        else {
//                         //  comm_data->done_flags[i] = 1;
//                         // // DMEM_CheckAllDoneFlag(dmem_all_data);
//                         //  if (dmem_all_data->comm.all_done_flag == 1){
//                         //     comm_data->done_flags[i] = 2;
//                         //     comm_data->data[i][vec_len] = 2.0;
//                         //  }
//                        }
//                     } 
//                     for (HYPRE_Int j = vec_start; j < vec_end; j++){
//                        dmem_all_data->comm.fine_send_data[j] =
//                           v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
//                     }
//                     comm_data->message_count[i]++;
//                     hypre_MPI_Isend(&(dmem_all_data->comm.fine_send_data[vec_start]),
//                                     vec_len,
//                                     HYPRE_MPI_REAL,
//                                     ip,
//                                     comm_data->tag,
//                                     MPI_COMM_WORLD,
//                                     &(comm_data->requests[i]));
//                  }
//               }
//            }
//            break;
//         /* fine outside recv */
//         case FINE_INTRA_OUTSIDE_RECV:
//            if (dmem_all_data->input.async_flag == 1){
//               if (comm_data->done_flags[i] < 2){
//                  do {
//                     flag = 0;
//                     hypre_MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
//                     if (flag){
//                        if (op == ACCUMULATE){
//                           for (HYPRE_Int j = vec_start; j < vec_end; j++){
//                              v[j] += dmem_all_data->comm.fine_recv_data[j];
//                           }
//                        }
//                        else if (op == READ){
//                           for (HYPRE_Int j = vec_start; j < vec_end; j++){
//                              v[j] = dmem_all_data->comm.fine_recv_data[j];
//                           }
//                        }
//                        comm_data->message_count[i]++;
//                        if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
//                           if (comm_data->message_count[i] == dmem_all_data->input.num_cycles){
//                              comm_data->done_flags[i] = 2;
//                              break;
//                           }
//                          // if (comm_data->data[i][vec_len] == 1.0){
//                          //    comm_data->done_flags[i] = 2;
//                          //    break;
//                          // }
//                        }
//                        else {
//                          // if (comm_data->data[i][vec_len] == 1.0){
//                          //    comm_data->done_flags[i] = 1;
//                          // }
//                          // else if (comm_data->data[i][vec_len] == 2.0){
//                          //    comm_data->done_flags[i] = 2;
//                          //    break;
//                          // }
//                        }
//                        hypre_MPI_Irecv(&(dmem_all_data->comm.fine_recv_data[vec_start]),
//                                        vec_len,
//                                        HYPRE_MPI_REAL,
//                                        ip,
//                                        comm_data->tag,
//                                        MPI_COMM_WORLD,
//                                        &(comm_data->requests[i]));
//                     }
//                  } while(flag);
//               }
//            }
//            else {
//               hypre_MPI_Irecv(&(dmem_all_data->comm.fine_recv_data[vec_start]),
//                               vec_len,
//                               HYPRE_MPI_REAL,
//                               ip,
//                               comm_data->tag,
//                               MPI_COMM_WORLD,
//                               &(comm_data->requests[i]));
//               comm_data->message_count[i]++;
//            }
//            break;
//      }
//   }
//}

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

   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
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
               hypre_MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
               if (flag){
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
                  if (comm_data->message_count[i] >= dmem_all_data->input.num_cycles-1 ||
                      dmem_all_data->iter.r_norm2_local_converge_flag == 1){
                     comm_data->data[i][vec_len] = 1.0;
                     if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
                        comm_data->done_flags[i] = 2;
                     }
                     else {
                        comm_data->done_flags[i] = 1;
                       // DMEM_CheckAllDoneFlag(dmem_all_data);
                        if (dmem_all_data->comm.all_done_flag == 1){
                           comm_data->done_flags[i] = 2;
                           comm_data->data[i][vec_len] = 2.0;
                        }
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
               do {
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
                     comm_data->message_count[i]++;
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
                     }
                     hypre_MPI_Irecv(comm_data->data[i],
                                     vec_len+1,
                                     HYPRE_MPI_REAL,
                                     ip,
                                     comm_data->tag,
                                     MPI_COMM_WORLD,
                                     &(comm_data->requests[i]));
                  }
               } while(flag);
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

  // if (comm_data->type == FINE_INTRA_INSIDE_RECV ||
  //     comm_data->type == FINE_INTRA_OUTSIDE_RECV){
  //    for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
  //       HYPRE_Int ip = comm_data->procs[i];
  //       HYPRE_Int vec_start = comm_data->start[i];
  //       HYPRE_Int vec_end = comm_data->end[i];
  //       HYPRE_Int vec_len = comm_data->len[i];
  //       if (op == ACCUMULATE){
  //          for (HYPRE_Int j = vec_start; j < vec_end; j++){
  //             v[j] += dmem_all_data->comm.fine_recv_data[j];
  //          }
  //       }
  //       else if (op == READ){
  //          for (HYPRE_Int j = vec_start; j < vec_end; j++){
  //             v[j] = dmem_all_data->comm.fine_recv_data[j];
  //          }
  //       }
  //    }
  // }
  // else if (comm_data->type == GRIDK_INSIDE_RECV || 
  //          comm_data->type == GRIDK_OUTSIDE_RECV){
  //    for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
  //       HYPRE_Int ip = comm_data->procs[i];
  //       HYPRE_Int vec_start = comm_data->start[i];
  //       HYPRE_Int vec_len = comm_data->len[i];
  //       if (op == ACCUMULATE){
  //          for (HYPRE_Int j = 0; j < vec_len; j++){
  //             v[vec_start+j] += comm_data->data[i][j];
  //          }
  //       }
  //       else if (op == READ){
  //          for (HYPRE_Int j = 0; j < vec_len; j++){
  //             v[vec_start+j] = comm_data->data[i][j];
  //          }
  //       }
  //    }
  // }
 
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
