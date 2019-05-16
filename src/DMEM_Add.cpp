#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Test.hpp"
#include "DMEM_Add.hpp"
#include "DMEM_SyncAdd.hpp"

void AddCycle(DMEM_AllData *dmem_all_data);
void FineSmooth(DMEM_AllData *dmem_all_data);
void AsyncRecvCleanup(DMEM_AllData *dmem_all_data);
void AsyncRecvStart(DMEM_AllData *dmem_all_data,
                    DMEM_CommData *comm_data);
void PrintMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);
HYPRE_Int CheckMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);
void AllOutsideRecv(DMEM_AllData *dmem_all_data);

void DMEM_Add(DMEM_AllData *dmem_all_data)
{
  // DMEM_TestResidual(dmem_all_data);
  // DMEM_TestCorrect(dmem_all_data);
  // DMEM_TestIsendrecv(dmem_all_data);
  // return;
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   hypre_ParVector *r;

   HYPRE_Int cycle = 1;
   HYPRE_Int num_cycles = dmem_all_data->input.num_cycles;
   HYPRE_Int start_cycle = dmem_all_data->input.start_cycle;
   HYPRE_Int increment_cycle = dmem_all_data->input.increment_cycle;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);

   if (dmem_all_data->input.async_flag == 1){
      AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.gridk_e_outside_recv));
      AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.gridk_r_outside_recv));
      AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.fine_outside_recv));
   }
   MPI_Barrier(MPI_COMM_WORLD);

   while (1){
      AddCycle(dmem_all_data);
     // FineSmooth(dmem_all_data);
     // DMEM_AddResidual(dmem_all_data);
      DMEM_AddCorrect(dmem_all_data);
      DMEM_AddResidual(dmem_all_data);
     
      if (dmem_all_data->input.async_flag == 1){
         if (CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_e_outside_send)) == 1 &&
             CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_r_outside_send)) == 1 &&
             CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.fine_outside_send))    == 1){
            break;
         }
      }
      else {
         r = dmem_all_data->vector_fine.r;
         HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r));
        // if (my_id == 0) printf("%d %e\n", cycle, res_norm/dmem_all_data->output.r0_norm2);
         if (cycle == num_cycles){
            break;
         }
      }
      cycle += increment_cycle;
   }
   MPI_Barrier(MPI_COMM_WORLD);
  // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_e_outside_recv));
  // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_r_outside_recv));
  // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.fine_outside_recv));
   if (dmem_all_data->input.async_flag == 1){
      AsyncRecvCleanup(dmem_all_data);
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         dmem_all_data->matrix.A_fine,
                                         dmem_all_data->vector_fine.x,
                                         1.0,
                                         dmem_all_data->vector_fine.b,
                                         dmem_all_data->vector_fine.r);
     // printf("%d\n", my_id);
      r = dmem_all_data->vector_fine.r;
      HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r));
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

void AddCycle(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   hypre_ParAMGData *amg_data = 
      (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;

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

   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   for (HYPRE_Int level = 0; level < my_grid; level++){
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;
      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
                                F_array[fine_grid],
                                0.0,
                                F_array[coarse_grid]);
      if (dmem_all_data->input.async_flag == 1){
         AllOutsideRecv(dmem_all_data);
      }
   }

   HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[my_grid]));
   HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[my_grid]));
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[my_grid]));
   f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[my_grid]));
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[my_grid]));
   for (HYPRE_Int i = 0; i < num_rows; i++){
      u_local_data[i] = dmem_all_data->input.smooth_weight * f_local_data[i] / A_data[A_i[i]];
   }
   hypre_ParCSRMatrixMatvec(1.0,
                            A_array[my_grid],
                            U_array[my_grid],
                            0.0,
                            Vtemp);
   for (HYPRE_Int i = 0; i < num_rows; i++){
      u_local_data[i] = 2.0 * u_local_data[i] -
         dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
   }
   
   if (dmem_all_data->input.async_flag == 1){
      AllOutsideRecv(dmem_all_data);
   }
   for (HYPRE_Int level = my_grid; level > 0; level--){
      HYPRE_Int fine_grid = level - 1;
      HYPRE_Int coarse_grid = level;
      hypre_ParCSRMatrixMatvec(1.0,
                               P_array[fine_grid], 
                               U_array[coarse_grid],
                               0.0,
                               U_array[fine_grid]);
      if (dmem_all_data->input.async_flag == 1){
         AllOutsideRecv(dmem_all_data);
      }
   }

  // if (my_grid == 0) hypre_ParVectorSetConstantValues(U_array[0], 0.0);

  // num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
  // u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p && my_grid == 1){
  //       for (HYPRE_Int i = 0; i < num_rows; i++){
  //          printf("%e\n", u_local_data[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void DMEM_AddCorrect(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

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
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));

   e = dmem_all_data->vector_fine.e;
   x = dmem_all_data->vector_fine.x;
   hypre_ParVectorSetConstantValues(e, 0.0);
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

  // HYPRE_Int my_grid = dmem_all_data->grid.my_grid;
  // if (my_grid == 0){
  //    hypre_ParVectorSetConstantValues(U_array[0], 0.0);
  // }

   hypre_MPI_Waitall(dmem_all_data->comm.gridk_e_inside_send.procs.size(),
                     dmem_all_data->comm.gridk_e_inside_send.requests,
                     MPI_STATUSES_IGNORE);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.gridk_e_outside_send.procs.size(),
                        dmem_all_data->comm.gridk_e_outside_send.requests,
                        MPI_STATUSES_IGNORE);
   }

   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_inside_send),
                 u_local_data,
                 ACCUMULATE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_outside_send),
                 u_local_data,
                 ACCUMULATE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_inside_recv),
                 NULL,
                 ACCUMULATE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_outside_recv),
                 e_local_data,
                 ACCUMULATE);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.gridk_e_inside_recv),
                e_local_data,
                ACCUMULATE);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.gridk_e_outside_recv),
                   e_local_data,
                   ACCUMULATE);
   }

   fine_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_fine[0]));
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

void DMEM_AddResidual(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int my_grid = dmem_all_data->grid.my_grid;

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

   hypre_MPI_Waitall(dmem_all_data->comm.fine_inside_send.procs.size(),
                     dmem_all_data->comm.fine_inside_send.requests,
                     MPI_STATUSES_IGNORE);
   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_inside_send),
                x_local_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.fine_outside_send.procs.size(),
                        dmem_all_data->comm.fine_outside_send.requests,
                        MPI_STATUSES_IGNORE);
   }
   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_outside_send),
                x_local_data,
                READ);
   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_inside_recv),
                NULL,
                READ);
   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_outside_recv),
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
                &(dmem_all_data->comm.fine_inside_recv),
                x_ghost_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.fine_outside_recv),
                   x_ghost_data,
                   READ);
   }

   hypre_CSRMatrixMatvec(-1.0,
                         offd,
                         x_ghost,
                         1.0,
                         r_local);

   HYPRE_Real *r_local_data = hypre_VectorData(r_local);
   amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));

   hypre_MPI_Waitall(dmem_all_data->comm.gridk_r_inside_send.procs.size(),
                     dmem_all_data->comm.gridk_r_inside_send.requests,
                     MPI_STATUSES_IGNORE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_inside_send),
                 r_local_data,
                 READ);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.gridk_r_outside_send.procs.size(),
                        dmem_all_data->comm.gridk_r_outside_send.requests,
                        MPI_STATUSES_IGNORE);
   }
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_outside_send),
                 r_local_data,
                 READ);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_inside_recv),
                 NULL,
                 READ);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_outside_recv),
                 f_local_data,
                 READ); 
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.gridk_r_inside_recv),
                f_local_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.gridk_r_outside_recv),
                   f_local_data,
                   READ);
   }
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
      if (comm_data->message_count[i] < dmem_all_data->input.num_cycles)
      printf("%d, %d, %d\n", comm_data->type, comm_data->message_count[i], dmem_all_data->input.num_cycles);
   }
}

HYPRE_Int CheckMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
      if (comm_data->message_count[i] < dmem_all_data->input.num_cycles){
         return 0;
      }
   }
   return 1;
}

void AsyncRecvStart(DMEM_AllData *dmem_all_data,
                    DMEM_CommData *comm_data)
{
   MPI_Status status;
   int flag;
   if (comm_data->type == FINE_OUTSIDE_RECV){
      for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
         HYPRE_Int ip = comm_data->procs[i];
         HYPRE_Int vec_start = comm_data->start[i];
         HYPRE_Int vec_len = comm_data->len[i];
         hypre_MPI_Irecv(&(dmem_all_data->comm.fine_recv_data[vec_start]),
                         vec_len,
                         HYPRE_MPI_REAL,
                         ip,
                         comm_data->tag,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
      }
   }
   else if (comm_data->type == GRIDK_OUTSIDE_RECV){
      for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
         HYPRE_Int ip = comm_data->procs[i];
         HYPRE_Int vec_len = comm_data->len[i];
         hypre_MPI_Irecv(comm_data->data[i],
                         vec_len,
                         HYPRE_MPI_REAL,
                         ip,
                         comm_data->tag,
                         MPI_COMM_WORLD,
                         &(comm_data->requests[i]));
      }
   }
}

void AsyncRecvCleanup(DMEM_AllData *dmem_all_data)
{
   hypre_ParAMGData *fine_amg_data;
   hypre_ParCSRMatrix **A_array_fine;
   hypre_ParVector *e, *x;
   HYPRE_Real *e_local_data, *x_local_data;
   HYPRE_Int fine_num_rows;

   e = dmem_all_data->vector_fine.e;
   x = dmem_all_data->vector_fine.x;
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   A_array_fine = hypre_ParAMGDataAArray(fine_amg_data);
   fine_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_fine[0]));

   while(1){
      if (CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_e_outside_recv)) == 1 &&
          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_r_outside_recv)) == 1 &&
          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.fine_outside_recv))    == 1){
         break;
      }

     // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_e_outside_recv));
     // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_r_outside_recv));
     // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.fine_outside_recv));
      
      hypre_ParVectorSetConstantValues(e, 0.0);
      GridkSendRecv(dmem_all_data,
                    &(dmem_all_data->comm.gridk_e_outside_recv),
                    e_local_data,
                    ACCUMULATE);
      for (HYPRE_Int i = 0; i < fine_num_rows; i++){
         x_local_data[i] += e_local_data[i];
      }

      GridkSendRecv(dmem_all_data,
                    &(dmem_all_data->comm.gridk_r_outside_recv),
                    NULL,
                    -1);
      FineSendRecv(dmem_all_data,
                   &(dmem_all_data->comm.fine_outside_recv),
                   NULL,
                   -1);
   }
   hypre_MPI_Waitall(dmem_all_data->comm.gridk_e_outside_send.procs.size(),
                     dmem_all_data->comm.gridk_e_outside_send.requests,
                     MPI_STATUSES_IGNORE);
   hypre_MPI_Waitall(dmem_all_data->comm.gridk_r_outside_send.procs.size(),
                     dmem_all_data->comm.gridk_r_outside_send.requests,
                     MPI_STATUSES_IGNORE);
   hypre_MPI_Waitall(dmem_all_data->comm.fine_outside_send.procs.size(),
                     dmem_all_data->comm.fine_outside_send.requests,
                     MPI_STATUSES_IGNORE);
}

void AllOutsideRecv(DMEM_AllData *dmem_all_data)
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);

   hypre_ParVector *e = dmem_all_data->vector_fine.e;
   hypre_ParVector *x = dmem_all_data->vector_fine.x;
   hypre_ParVectorSetConstantValues(e, 0.0);
   HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_outside_recv),
                 e_local_data,
                 ACCUMULATE);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
   for (HYPRE_Int i = 0; i < num_rows; i++){
      x_local_data[i] += e_local_data[i];
   }

   hypre_Vector *x_ghost = dmem_all_data->vector_fine.x_ghost;
   HYPRE_Real *x_ghost_data = hypre_VectorData(x_ghost);
   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_outside_recv),
                x_ghost_data,
                READ);
   
   amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0])); 
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_outside_recv),
                 f_local_data,
                 READ);
}

//HYPRE_Int DMEM_CheckConverge(DMEM_AllData *dmem_all_data, HYPRE_Int cycle)
//{
//   HYPRE_Int num_cycles = dmem_all_data->input.num_cycles;
//   if (dmem_all_data->input.async_flag == 1){
//      if (CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_e_inside_send))  == 1 &&
//          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_e_outside_send)) == 1 &&
//          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_r_inside_send))  == 1 &&
//          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridk_r_outside_send)) == 1 &&
//          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.fine_inside_send))     == 1 &&
//          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.fine_outside_send))    == 1){
//         return 1;
//      }
//      else {
//         return 0;
//      }
//   }
//   else {
//      if (cycle == dmem_all_data->input.num_cycles){
//         return 1;
//      }
//      else {
//         return 0;
//      }
//   }
//}
