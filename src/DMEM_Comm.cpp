#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "Misc.hpp"

using namespace std;

void FineSendRecv(DMEM_AllData *dmem_all_data,
                  DMEM_CommData *comm_data,
                  HYPRE_Real *v)
{
   int flag;
   hypre_ParCSRMatrix *A = dmem_all_data->matrix.A;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      HYPRE_Int ip = comm_data->procs[i];
      HYPRE_Int vec_start = comm_data->start[i];
      HYPRE_Int vec_end = comm_data->end[i];
      HYPRE_Int vec_len = comm_data->len[i];
      switch(comm_data->type){
	 /* fine inside send */
         case FINE_INSIDE_SEND:
            for (HYPRE_Int j = vec_start; j < vec_end; j++){
               dmem_all_data->comm.fine_send_data[j] = 
                  v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
            MPI_Isend(&(dmem_all_data->comm.fine_send_data[vec_start]),
                      vec_len,
                      MPI_DOUBLE,
                      ip,
                      0,
                      MPI_COMM_WORLD,
                      &(comm_data->requests[i]));
            break;
         /* fine inside recv */
         case FINE_INSIDE_RECV:
            MPI_Irecv(&(dmem_all_data->comm.fine_recv_data[vec_start]),
                      vec_len,
                      MPI_DOUBLE,
                      ip,
                      0,
                      MPI_COMM_WORLD,
                      &(comm_data->requests[i]));
            break;
         /* fine outside send */
         case FINE_OUTSIDE_SEND:
            MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
            if (flag){
               for (HYPRE_Int j = vec_start; j < vec_end; j++){
                  dmem_all_data->comm.fine_send_data[j] =
                     v[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
               }
               MPI_Isend(&(dmem_all_data->comm.fine_send_data[vec_start]),
                         vec_len,
                         MPI_DOUBLE,
                         ip,
                         0,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
            }
            break;
         /* fine outside recv */
         case FINE_OUTSIDE_RECV:
            MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
            if (flag){
               for (HYPRE_Int j = vec_start; j < vec_end; j++){
                  v[j] = dmem_all_data->comm.fine_recv_data[j];
               }
               MPI_Irecv(&(dmem_all_data->comm.fine_recv_data[vec_start]),
                         vec_len,
                         MPI_DOUBLE,
                         ip,
                         0,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
            }
            break;
      }
   }
}

void GridkSendRecv(DMEM_AllData *dmem_all_data,
                   DMEM_CommData *comm_data,
		   HYPRE_Real *v)
{
   int flag;
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      HYPRE_Int ip = comm_data->procs[i];
      HYPRE_Int vec_start = comm_data->start[i];
      HYPRE_Int vec_len = comm_data->len[i];
      switch(comm_data->type){
	 /* gridk inside send */
         case GRIDK_INSIDE_SEND:
            for (HYPRE_Int j = 0; j < vec_len; j++){
               comm_data->data[i][j] = v[vec_start+j];
            }
            MPI_Isend(comm_data->data[i],
                      vec_len,
                      MPI_DOUBLE,
                      ip,
                      0,
                      MPI_COMM_WORLD,
                      &(comm_data->requests[i]));
            break;
         /* gridk inside recv */
         case GRIDK_INSIDE_RECV:
            MPI_Irecv(comm_data->data[i],
                      vec_len,
                      MPI_DOUBLE,
                      ip,
                      0,
                      MPI_COMM_WORLD,
                      &(comm_data->requests[i]));
            break;
         /* gridk outside send */
         case GRIDK_OUTSIDE_SEND:
            MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
            if (flag){
	       for (HYPRE_Int j = 0; j < vec_len; j++){
                  comm_data->data[i][j] = v[vec_start+j];
               }
               MPI_Isend(comm_data->data[i],
                         vec_len,
                         MPI_DOUBLE,
                         ip,
                         0,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
            }
            break;
         /* gridk outside recv */
         case GRIDK_OUTSIDE_RECV:
            MPI_Test(&(comm_data->requests[i]), &flag, MPI_STATUS_IGNORE);
            if (flag){
               for (HYPRE_Int j = 0; j < vec_len; j++){
                  v[vec_start+j] = comm_data->data[i][j];
               }
               MPI_Irecv(comm_data->data[i],
                         vec_len,
                         MPI_DOUBLE,
                         ip,
                         0,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
            }
            break;
      }
   }
}


void CompleteRecv(DMEM_AllData *dmem_all_data,
                  DMEM_CommData *comm_data,
                  HYPRE_Real *v)
{
   MPI_Waitall(comm_data->procs.size(),
               comm_data->requests,
               MPI_STATUSES_IGNORE);

   if (comm_data->type == FINE_INSIDE_RECV ||
       comm_data->type == FINE_OUTSIDE_RECV){
      for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
         HYPRE_Int ip = comm_data->procs[i];
         HYPRE_Int vec_start = comm_data->start[i];
         HYPRE_Int vec_end = comm_data->end[i];
         HYPRE_Int vec_len = comm_data->len[i];
         for (HYPRE_Int j = vec_start; j < vec_end; j++){
            v[j] = dmem_all_data->comm.fine_recv_data[j];
         }
      }
   }
   else if (comm_data->type == GRIDK_INSIDE_RECV || 
            comm_data->type == GRIDK_OUTSIDE_RECV){
      for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
         HYPRE_Int ip = comm_data->procs[i];
         HYPRE_Int vec_start = comm_data->start[i];
         HYPRE_Int vec_len = comm_data->len[i];
         for (HYPRE_Int j = 0; j < vec_len; j++){
            v[vec_start+j] = comm_data->data[i][j];
         }
      }
   }
}
