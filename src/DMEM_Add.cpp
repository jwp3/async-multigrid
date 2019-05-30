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
void PrintMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);
HYPRE_Int CheckMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);
void AsyncStart(DMEM_AllData *dmem_all_data);
void AsyncEnd(DMEM_AllData *dmem_all_data);
int DMEM_CheckAsyncConverge(DMEM_AllData *dmem_all_data, HYPRE_Int cycle);
void AllOutsideRecv(DMEM_AllData *dmem_all_data);

void DMEM_Add(DMEM_AllData *dmem_all_data)
{
  // DMEM_TestCorrect_LocalRes(dmem_all_data);
  // return;
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   hypre_ParVector *r;

   HYPRE_Int cycle = 1;
   HYPRE_Int num_cycles = dmem_all_data->input.num_cycles;
   HYPRE_Int start_cycle = dmem_all_data->input.start_cycle;
   HYPRE_Int increment_cycle = dmem_all_data->input.increment_cycle;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParAMGData *amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);
   
   if (dmem_all_data->input.res_compute_type == LOCAL){
      DMEM_ResidualToGridk_LocalRes(dmem_all_data, dmem_all_data->vector_fine.b, F_array_gridk[0]);
   }

   //TODO: modify for local-res
   if (dmem_all_data->input.async_flag == 1){
      AsyncStart(dmem_all_data);
   }

   while (1){
      AddCycle(dmem_all_data);
     // FineSmooth(dmem_all_data);
     // DMEM_AddResidual_GlobalRes(dmem_all_data);
      if (dmem_all_data->input.res_compute_type == GLOBAL){
         DMEM_AddCorrect_GlobalRes(dmem_all_data);
         DMEM_AddResidual_GlobalRes(dmem_all_data);
      }
      else {
         DMEM_AddCorrect_LocalRes(dmem_all_data);
         DMEM_AddResidual_LocalRes(dmem_all_data);
      }
     
      if (dmem_all_data->input.async_flag == 1){
         if (DMEM_CheckAsyncConverge(dmem_all_data, cycle) == 1){
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
  // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv));
  // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv));
  // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));

   if (dmem_all_data->input.async_flag == 1){
      AsyncEnd(dmem_all_data);
   }
 
   if (dmem_all_data->input.res_compute_type == LOCAL){
      //TODO: modify for local-res 
      DMEM_SolutionToFinest_LocalRes(dmem_all_data, dmem_all_data->vector_gridk.x, dmem_all_data->vector_fine.x);
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         dmem_all_data->matrix.A_fine,
                                         dmem_all_data->vector_fine.x,
                                         1.0,
                                         dmem_all_data->vector_fine.b,
                                         dmem_all_data->vector_fine.r);
      r = dmem_all_data->vector_fine.r;
      HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r));
   }
   else {
      if (dmem_all_data->input.async_flag == 1){
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
  // f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p && my_grid == 1){
  //       for (HYPRE_Int i = 0; i < num_rows; i++){
  //          printf("%e\n", f_local_data[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void DMEM_AddCorrect_LocalRes(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   hypre_ParAMGData *gridk_amg_data;
   hypre_ParCSRMatrix **A_array_gridk;
   hypre_ParVector **U_array;
   hypre_ParVector *e, *x;
   HYPRE_Real *e_local_data, *x_local_data, *u_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   A_array_gridk = hypre_ParAMGDataAArray(gridk_amg_data);

   U_array = hypre_ParAMGDataUArray(gridk_amg_data);
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));

   e = dmem_all_data->vector_gridk.e;
   x = dmem_all_data->vector_gridk.x;
   hypre_ParVectorSetConstantValues(e, 0.0);
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   hypre_MPI_Waitall(dmem_all_data->comm.gridjToGridk_Correct_insideSend.procs.size(),
                     dmem_all_data->comm.gridjToGridk_Correct_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(),
                        dmem_all_data->comm.gridjToGridk_Correct_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }

   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridjToGridk_Correct_insideSend),
                 u_local_data,
                 ACCUMULATE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridjToGridk_Correct_insideRecv),
                 NULL,
                 ACCUMULATE);

   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend),
                 u_local_data,
                 ACCUMULATE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                 e_local_data,
                 ACCUMULATE);

   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.gridjToGridk_Correct_insideRecv),
                e_local_data,
                ACCUMULATE);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                   e_local_data,
                   ACCUMULATE);
   }

   fine_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_gridk[0]));
   for (HYPRE_Int i = 0; i < fine_num_rows; i++){
      x_local_data[i] += e_local_data[i];
     // printf("%e\n", x_local_data[i]);
   }
}

void DMEM_AddResidual_LocalRes(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data_gridk =
      (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);
   hypre_ParCSRMatrix **A_array_gridk = hypre_ParAMGDataAArray(amg_data_gridk);

   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                      A_array_gridk[0],
                                      dmem_all_data->vector_gridk.x,
                                      1.0,
                                      dmem_all_data->vector_gridk.b,
                                      F_array_gridk[0]);
}

//TODO: communicate locally within grid
void DMEM_SolutionToFinest_LocalRes(DMEM_AllData *dmem_all_data,
                                    hypre_ParVector *x_gridk,
                                    hypre_ParVector *x_fine)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   HYPRE_Real *x_gridk_local_data, *x_fine_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   x_gridk_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x_gridk));
   x_fine_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x_fine));

   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Correct_insideSend.requests,
                     MPI_STATUSES_IGNORE);

   GridkSendRecv(dmem_all_data,
                       &(dmem_all_data->comm.finestToGridk_Correct_insideSend),
                       x_gridk_local_data,
                       READ);
   GridkSendRecv(dmem_all_data,
                       &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                       NULL,
                       READ);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                x_fine_local_data,
                READ);
}

void DMEM_ResidualToGridk_LocalRes(DMEM_AllData *dmem_all_data,
                                   hypre_ParVector *r,
                                   hypre_ParVector *f)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   HYPRE_Real *r_local_data, *f_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(r));
   f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(f));

   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Residual_insideSend.requests,
                     MPI_STATUSES_IGNORE);

   GridkSendRecv(dmem_all_data,
                       &(dmem_all_data->comm.finestToGridk_Residual_insideSend),
                       r_local_data,
                       READ);
   GridkSendRecv(dmem_all_data,
                       &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
                       NULL,
                       READ);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
                f_local_data,
                READ);
}

void DMEM_AddCorrect_GlobalRes(DMEM_AllData *dmem_all_data)
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

   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Correct_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_outsideSend.procs.size(),
                        dmem_all_data->comm.finestToGridk_Correct_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }

   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Correct_insideSend),
                 u_local_data,
                 ACCUMULATE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Correct_outsideSend),
                 u_local_data,
                 ACCUMULATE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                 NULL,
                 ACCUMULATE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv),
                 e_local_data,
                 ACCUMULATE);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                e_local_data,
                ACCUMULATE);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv),
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

void DMEM_AddResidual_GlobalRes(DMEM_AllData *dmem_all_data)
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

   hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_insideSend.procs.size(),
                     dmem_all_data->comm.finestIntra_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   FineIntraSendRecv(dmem_all_data,
                &(dmem_all_data->comm.finestIntra_insideSend),
                x_local_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                        dmem_all_data->comm.finestIntra_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }
   FineIntraSendRecv(dmem_all_data,
                &(dmem_all_data->comm.finestIntra_outsideSend),
                x_local_data,
                READ);
   FineIntraSendRecv(dmem_all_data,
                &(dmem_all_data->comm.finestIntra_insideRecv),
                NULL,
                READ);
   FineIntraSendRecv(dmem_all_data,
                &(dmem_all_data->comm.finestIntra_outsideRecv),
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
                &(dmem_all_data->comm.finestIntra_insideRecv),
                x_ghost_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.finestIntra_outsideRecv),
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

   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Residual_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Residual_insideSend),
                 r_local_data,
                 READ);
   if (dmem_all_data->input.async_flag == 0){
      hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_outsideSend.procs.size(),
                        dmem_all_data->comm.finestToGridk_Residual_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Residual_outsideSend),
                 r_local_data,
                 READ);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
                 NULL,
                 READ);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv),
                 f_local_data,
                 READ); 
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
                f_local_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv),
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
   if (comm_data->type == FINE_INTRA_OUTSIDE_RECV){
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

void AsyncStart(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->input.async_type == GLOBAL){
      AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv));
      AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv));
      AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));
   }
   else {
      AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv));
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
      if (dmem_all_data->input.async_type == GLOBAL){
         if (CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv)) == 1 &&
             CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv)) == 1 &&
             CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv))    == 1){
            break;
         }
      }
      else {
         if (CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv))){
            break;
         }
      }

     // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv));
     // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv));
     // PrintMessageCount(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));
      
      hypre_ParVectorSetConstantValues(e, 0.0);
      GridkSendRecv(dmem_all_data,
                    &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv),
                    e_local_data,
                    ACCUMULATE);
      for (HYPRE_Int i = 0; i < fine_num_rows; i++){
         x_local_data[i] += e_local_data[i];
      }

      if (dmem_all_data->input.async_type == GLOBAL){
         GridkSendRecv(dmem_all_data,
                       &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv),
                       NULL,
                       -1);
         FineIntraSendRecv(dmem_all_data,
                      &(dmem_all_data->comm.finestIntra_outsideRecv),
                      NULL,
                      -1);
      }
   }
   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_outsideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Correct_outsideSend.requests,
                     MPI_STATUSES_IGNORE);
   if (dmem_all_data->input.async_type == GLOBAL){
      hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_outsideSend.procs.size(),
                        dmem_all_data->comm.finestToGridk_Residual_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
      hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                        dmem_all_data->comm.finestIntra_outsideSend.requests,
                        MPI_STATUSES_IGNORE);
   }
}

void AsyncEnd(DMEM_AllData *dmem_all_data)
{
   AsyncRecvCleanup(dmem_all_data);
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
                 &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv),
                 e_local_data,
                 ACCUMULATE);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
   for (HYPRE_Int i = 0; i < num_rows; i++){
      x_local_data[i] += e_local_data[i];
   }

   if (dmem_all_data->input.async_type == GLOBAL){
      hypre_Vector *x_ghost = dmem_all_data->vector_fine.x_ghost;
      HYPRE_Real *x_ghost_data = hypre_VectorData(x_ghost);
      FineIntraSendRecv(dmem_all_data,
                   &(dmem_all_data->comm.finestIntra_outsideRecv),
                   x_ghost_data,
                   READ);
      
      amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
      hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
      HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0])); 
      GridkSendRecv(dmem_all_data,
                    &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv),
                    f_local_data,
                    READ);
   }
}

int DMEM_CheckAsyncConverge(DMEM_AllData *dmem_all_data, HYPRE_Int cycle)
{
   HYPRE_Int num_cycles = dmem_all_data->input.num_cycles;
   if (dmem_all_data->input.async_type == GLOBAL){
      if (CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideSend)) == 1 &&
          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideSend)) == 1 &&
          CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend)) == 1){
         return 1;
      }
   }
   else {
      if (CheckMessageCount(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend)) == 1){
         return 1; 
      }
   }
   
   return 0;
}
