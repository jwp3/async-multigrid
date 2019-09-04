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

void DMEM_AsyncSmooth(DMEM_AllData *dmem_all_data, int level)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
   HYPRE_Real *v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   HYPRE_Real *Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);
   HYPRE_Real *x_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost);

   if (dmem_all_data->input.async_flag == 1){
      dmem_all_data->comm.outside_recv_done_flag = 0;
      dmem_all_data->comm.async_smooth_done_flag = 0;
      DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));
   }

   while (1){
      if (dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL){
      }
      else {
         for (int i = 0; i < num_rows; i++){
            u_local_data[i] = dmem_all_data->input.smooth_weight * r_local_data[i] / A_diag_data[A_diag_i[i]];
            x_local_data[i] += u_local_data[i];
         }
      }

      if (dmem_all_data->input.async_flag == 1){
         DMEM_AddCheckComm(dmem_all_data);
      }
      AsyncSmoothAddCorrect_LocalRes(dmem_all_data);
      if (dmem_all_data->input.async_flag == 1){
         DMEM_AddCheckComm(dmem_all_data);
      }
      else {
         hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                           dmem_all_data->comm.finestIntra_outsideSend.requests,
                           MPI_STATUSES_IGNORE);
      }
      int converge_flag = AsyncSmoothCheckConverge(dmem_all_data);
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.finestIntra_outsideSend),
               x_local_data,
               WRITE);
      if (dmem_all_data->input.async_flag == 1){
         DMEM_AddCheckComm(dmem_all_data);
      }
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.finestIntra_outsideRecv),
               x_ghost_data,
               READ); 
      if (dmem_all_data->input.async_flag == 1){
         DMEM_AddCheckComm(dmem_all_data);
      }
      else {
         CompleteRecv(dmem_all_data,
                      &(dmem_all_data->comm.finestIntra_outsideRecv),
                      x_ghost_data,
                      READ);
         hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                           dmem_all_data->comm.finestIntra_outsideSend.requests,
                           MPI_STATUSES_IGNORE);
      }

      if (dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL){
      }
      else {
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
      }

      if (dmem_all_data->input.async_flag == 0){
         double residual_norm_begin = MPI_Wtime();
         hypre_Vector *r = hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.r);
         dmem_all_data->iter.r_norm2_local =
            sqrt(InnerProd(r, r, dmem_all_data->grid.my_comm))/dmem_all_data->output.r0_norm2;
         dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
         if (dmem_all_data->iter.r_norm2_local < dmem_all_data->input.tol){
            dmem_all_data->iter.r_norm2_local_converge_flag = 1;
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
   if (dmem_all_data->input.async_flag == 1){
      AsyncSmoothEnd(dmem_all_data);
   }
   hypre_ParVectorCopy(dmem_all_data->vector_gridk.r, F_array[level]);
   dmem_all_data->comm.async_smooth_done_flag = 1;
}

int AsyncSmoothCheckConverge(DMEM_AllData *dmem_all_data)
{
   double begin = MPI_Wtime();
   if (dmem_all_data->input.async_flag == 1){
      if (DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv), 0)){
         dmem_all_data->comm.outside_recv_done_flag = 1;
         return 1;
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

      if (cycle >= num_cycles-1 || dmem_all_data->iter.r_norm2_local_converge_flag == 1){
         return 1;
      }
   }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
   return 0;
}

void AsyncSmoothAddCorrect_LocalRes(DMEM_AllData *dmem_all_data)
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
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(),
                        dmem_all_data->comm.gridjToGridk_Correct_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }
   dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - begin;
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend),
            u_local_data,
            ACCUMULATE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
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
   for (int i = 0; i < num_rows; i++){
      x_local_data[i] += e_local_data[i];
   }
   dmem_all_data->output.correct_wtime += MPI_Wtime() - begin;
   if (dmem_all_data->input.async_flag == 1){
      DMEM_AddCheckComm(dmem_all_data);
   }
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

      hypre_ParVectorSetConstantValues(e, 0.0);
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
      for (HYPRE_Int i = 0; i < num_rows; i++){
         x_local_data[i] += e_local_data[i];
      }
   }


   begin = MPI_Wtime();
   CompleteInFlight(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend));
   dmem_all_data->output.comm_wtime += MPI_Wtime() - begin;
}

void AsyncSmoothEnd(DMEM_AllData *dmem_all_data)
{
   AsyncSmoothRecvCleanup(dmem_all_data);
}

void DMEM_AddSmooth(DMEM_AllData *dmem_all_data, int coarsest_level)
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

   HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
   HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
   num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
   f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[coarsest_level]));
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[coarsest_level]));

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
      hypre_ParCSRMatrixMatvec(1.0,
                               A_array[coarsest_level],
                               U_array[coarsest_level],
                               0.0,
                               Vtemp);
      if (dmem_all_data->input.smoother == L1_JACOBI){
         for (int i = 0; i < num_rows; i++){
            u_local_data[i] = 2.0 * u_local_data[i] - v_local_data[i] / dmem_all_data->matrix.L1_row_norm_gridk[coarsest_level][i];
         }
      }
      else {
         for (int i = 0; i < num_rows; i++){
            u_local_data[i] = 2.0 * u_local_data[i] - dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
         }
      }
   }
}
