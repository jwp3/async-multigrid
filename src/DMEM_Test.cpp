#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Add.hpp"

void DMEM_TestCorrect_LocalRes(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector **U_array;
   hypre_ParVector *u, *e, *x, *Vtemp;
   HYPRE_Real *e_local_data, *x_local_data, *u_local_data, *v_local_data;
   HYPRE_Int fine_num_rows;

   amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   A_array = hypre_ParAMGDataAArray(amg_data);
   fine_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));

   e = dmem_all_data->vector_gridk.e;
   x = dmem_all_data->vector_gridk.x;
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   U_array = hypre_ParAMGDataUArray(amg_data);

   HYPRE_Int num_cycles = dmem_all_data->input.num_cycles;
   HYPRE_Int start_cycle = dmem_all_data->input.start_cycle;
   HYPRE_Int increment_cycle = dmem_all_data->input.increment_cycle;

   hypre_ParVectorSetConstantValues(x, 0.0);

   Vtemp = hypre_ParAMGDataVtemp(amg_data);
   hypre_ParVectorSetConstantValues(Vtemp, 0.0);
   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
   HYPRE_Real alpha = 1.0;
   hypre_ParVectorSetConstantValues(U_array[0], alpha);

   for (HYPRE_Int cycle = start_cycle; cycle <= num_cycles; cycle += increment_cycle){
      hypre_ParVectorSetConstantValues(e, 0.0);
      DMEM_AddCorrect_LocalRes(dmem_all_data);

      MPI_Barrier(MPI_COMM_WORLD);
      if (my_id == 0) printf("cycle %d\n", cycle);
      MPI_Barrier(MPI_COMM_WORLD);
      for (HYPRE_Int i = 0; i < fine_num_rows; i++){
         v_local_data[i] += dmem_all_data->grid.num_levels;
         HYPRE_Real c = fabs(x_local_data[i] - v_local_data[i])/fmax(x_local_data[i], v_local_data[i]);
         if (c > 1e-16) printf("correct failed: %e, %e\n", x_local_data[i], v_local_data[i]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
}

void DMEM_TestIsendrecv(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   HYPRE_Int flag;
   MPI_Status status;
   MPI_Request request = MPI_REQUEST_NULL;
   HYPRE_Int message_size = 10000, num_messages = 1000;
   HYPRE_Real **sendbuf = hypre_CTAlloc(HYPRE_Real *, message_size, dmem_all_data->input.hypre_memory);
   HYPRE_Real *recvbuf = hypre_CTAlloc(HYPRE_Real, message_size, dmem_all_data->input.hypre_memory);
   MPI_Request *send_requests = (MPI_Request *)calloc(num_messages, sizeof(MPI_Request));
   vector<double> recv_vec;

   if (my_id == 0){
      for (int i = 0; i < num_messages; i++){
         sendbuf[i] = hypre_CTAlloc(HYPRE_Real, message_size, dmem_all_data->input.hypre_memory);
         sendbuf[i][0] = i;
         hypre_MPI_Isend(sendbuf[i],
                         message_size,
                         HYPRE_MPI_REAL,
                         1,
                         0,
                         MPI_COMM_WORLD,
                         &(send_requests[i]));
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);
   if (my_id == 0){
      hypre_MPI_Waitall(num_messages,
                        send_requests,
                        MPI_STATUSES_IGNORE);
   }
   else if (my_id == 1){
      HYPRE_Int count_recv = 1;
      hypre_MPI_Irecv(recvbuf,
                      message_size,
                      HYPRE_MPI_REAL,
                      0,
                      0,
                      MPI_COMM_WORLD,
                      &(request));
      while(1){
         hypre_MPI_Test(&request, &flag, &status);
         if (flag){
            count_recv++;
            recv_vec.push_back(recvbuf[0]);
            if (count_recv == num_messages+1) break;
            hypre_MPI_Irecv(recvbuf,
                            message_size,
                            HYPRE_MPI_REAL,
                            0,
                            0,
                            MPI_COMM_WORLD,
                            &(request));
         }
      }
      for (int i = 1; i < num_messages; i++){
         if (recv_vec[i-1] > recv_vec[i]) printf("%f\n", recv_vec[i]);
      }
     // for (int i = 0; i < num_messages; i++){
     //    hypre_MPI_Recv(recvbuf,
     //                   message_size,
     //                   HYPRE_MPI_REAL,
     //                   0,
     //                   0,
     //                   MPI_COMM_WORLD,
     //                   MPI_STATUS_IGNORE);
     // }
   }
}
