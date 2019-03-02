#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"

void AddCycle(DMEM_AllData *dmem_all_data);
void AddCorrect(DMEM_AllData *dmem_all_data);
void AddResidual(DMEM_AllData *dmem_all_data);

void DMEM_Add(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   HYPRE_Int num_cycles = dmem_all_data->input.num_cycles;
   HYPRE_Int start_cycle = dmem_all_data->input.start_cycle;
   HYPRE_Int increment_cycle = dmem_all_data->input.increment_cycle;

   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   for (HYPRE_Int cycle = start_cycle; cycle < num_cycles; cycle += increment_cycle){
      AddCycle(dmem_all_data);
      AddCorrect(dmem_all_data);
      AddResidual(dmem_all_data);
   }
   MPI_Barrier(MPI_COMM_WORLD);
  // printf("%d\n", my_id);
}

void AddCycle(DMEM_AllData *dmem_all_data)
{
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

   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParVectorCopy(dmem_all_data->vector_fine.r,
		       F_array[0]);

   for (HYPRE_Int level = 0; level < my_grid; level++){
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;
      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
				F_array[fine_grid],
                                0.0,
                                F_array[coarse_grid]);
   }


   HYPRE_Int *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   HYPRE_Int rlx_coarse = grid_relax_type[3];
  // if (my_grid < num_levels-1){
      HYPRE_Real *A_data = 
         hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[my_grid]));
      HYPRE_Int *A_i = 
         hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[my_grid]));
      HYPRE_Int num_rows = 
         hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[my_grid]));
      f_local_data = 
         hypre_VectorData(hypre_ParVectorLocalVector(F_array[my_grid]));
      u_local_data = 
         hypre_VectorData(hypre_ParVectorLocalVector(U_array[my_grid]));
      v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

      for (HYPRE_Int i = 0; i < num_rows; i++){
         u_local_data[i] = 
            dmem_all_data->input.smooth_weight * f_local_data[i] / A_data[A_i[i]];
      }
     // hypre_ParCSRMatrixMatvec(1.0,
     //                          A_array[my_grid],
     //                          U_array[my_grid],
     //                          0.0,
     //                          Vtemp);
     // for (HYPRE_Int i = 0; i < num_rows; i++){
     //    u_local_data[i] = 2.0 * u_local_data[i] -
     //       dmem_all_data->input.smooth_weight * v_local_data[i] / A_data[A_i[i]];
     // }
  // }
  // else {
  //    hypre_GaussElimSolve(amg_data, my_grid, 99);
  // }
   
   for (HYPRE_Int level = my_grid-1; level >= 0; level--){
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;
      hypre_ParCSRMatrixMatvec(1.0,
                               P_array[fine_grid], 
                               U_array[coarse_grid],
                               0.0,
                               U_array[fine_grid]);            
   }
}

void AddCorrect(DMEM_AllData *dmem_all_data)
{
   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);

   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));

   hypre_ParVector *e = dmem_all_data->vector_fine.e;
   hypre_ParVector *x = dmem_all_data->vector_fine.x;
   HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_inside_send),
                 u_local_data);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_outside_send),
                 u_local_data);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_inside_recv),
                 NULL);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_e_outside_recv),
                 e_local_data);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.gridk_e_inside_recv),
                e_local_data);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.gridk_e_outside_recv),
                   e_local_data);
   }
   MPI_Waitall(dmem_all_data->comm.gridk_e_inside_send.procs.size(),
               dmem_all_data->comm.gridk_e_inside_send.requests,
               MPI_STATUSES_IGNORE);
   if (dmem_all_data->input.async_flag == 0){
      MPI_Waitall(dmem_all_data->comm.gridk_e_outside_send.procs.size(),
                  dmem_all_data->comm.gridk_e_outside_send.requests,
                  MPI_STATUSES_IGNORE);
   }
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
   for (HYPRE_Int i = 0; i < num_rows; i++){
      x_local_data[i] += e_local_data[i];
   }
}

void AddResidual(DMEM_AllData *dmem_all_data)
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

   hypre_Vector *x_ghost = hypre_SeqVectorCreate(num_cols_offd);
   hypre_SeqVectorInitialize(x_ghost);

   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_inside_send),
                hypre_VectorData(x_local));
   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_inside_recv),
                NULL);
   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_outside_send),
                hypre_VectorData(x_local));
   FineSendRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_outside_recv),
                hypre_VectorData(x_ghost));

   hypre_CSRMatrixMatvecOutOfPlace(-1.0,
                                   diag,
                                   x_local,
                                   1.0,
                                   b_local,
                                   r_local,
                                   0);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.fine_inside_recv),
                hypre_VectorData(x_ghost));
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.fine_outside_recv),
                   hypre_VectorData(x_ghost));
   }

   hypre_CSRMatrixMatvec(-1.0,
                         offd,
                         x_ghost,
                         1.0,
                         r_local);
   MPI_Waitall(dmem_all_data->comm.fine_inside_send.procs.size(),
               dmem_all_data->comm.fine_inside_send.requests,
               MPI_STATUSES_IGNORE); 
   if (dmem_all_data->input.async_flag == 0){
      MPI_Waitall(dmem_all_data->comm.fine_outside_send.procs.size(),
                  dmem_all_data->comm.fine_outside_send.requests,
                  MPI_STATUSES_IGNORE);
   }

   HYPRE_Real *r_local_data = hypre_VectorData(r_local);
   amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));

   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_inside_send),
                 r_local_data);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_outside_send),
                 r_local_data);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_inside_recv),
                 NULL);
   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_outside_recv),
                 f_local_data); 
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.gridk_r_inside_recv),
                f_local_data);
   if (dmem_all_data->input.async_flag == 0){
      CompleteRecv(dmem_all_data,
                   &(dmem_all_data->comm.gridk_r_outside_recv),
                   f_local_data);
   }
   MPI_Waitall(dmem_all_data->comm.gridk_r_inside_send.procs.size(),
               dmem_all_data->comm.gridk_r_inside_send.requests,
               MPI_STATUSES_IGNORE);
   if (dmem_all_data->input.async_flag == 0){
      MPI_Waitall(dmem_all_data->comm.gridk_r_outside_send.procs.size(),
                  dmem_all_data->comm.gridk_r_outside_send.requests,
                  MPI_STATUSES_IGNORE);
   }

}
