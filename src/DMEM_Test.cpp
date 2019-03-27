#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Add.hpp"

void DMEM_TestCorrect(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   hypre_ParCSRMatrix **A_array_fine, **A_array_gridk;
   hypre_ParVector **U_array;
   hypre_ParVector *u, *e, *x, *Vtemp;
   HYPRE_Real *e_local_data, *x_local_data, *u_local_data, *v_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   A_array_fine = hypre_ParAMGDataAArray(fine_amg_data);
   fine_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_fine[0]));

   e = dmem_all_data->vector_fine.e;
   x = dmem_all_data->vector_fine.x;
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   U_array = hypre_ParAMGDataUArray(gridk_amg_data);

   HYPRE_Int num_cycles = dmem_all_data->input.num_cycles;
   HYPRE_Int start_cycle = dmem_all_data->input.start_cycle;
   HYPRE_Int increment_cycle = dmem_all_data->input.increment_cycle;

   hypre_ParVectorSetConstantValues(x, 0.0);

   Vtemp = hypre_ParAMGDataVtemp(fine_amg_data);
   hypre_ParVectorSetConstantValues(Vtemp, 0.0);
   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
   HYPRE_Real alpha = 1.0;
   hypre_ParVectorSetConstantValues(U_array[0], alpha);

   for (HYPRE_Int cycle = start_cycle; cycle <= num_cycles; cycle += increment_cycle){
      hypre_ParVectorSetConstantValues(e, 0.0);
      DMEM_AddCorrect(dmem_all_data);

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

void DMEM_TestResidual(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData *)dmem_all_data->hypre.solver;

   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   HYPRE_Int fine_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
   HYPRE_Int fine_first_row = hypre_ParCSRMatrixFirstRowIndex(A_array[0]);

   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
   HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_fine.x));
   for (HYPRE_Int i = 0; i < fine_num_rows; i++){
      x_local_data[i] = RandDouble(-1.0, 1.0);
   }

   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                      A_array[0],
                                      dmem_all_data->vector_fine.x,
                                      1.0,
                                      dmem_all_data->vector_fine.b,
                                      Vtemp);
   DMEM_AddResidual(dmem_all_data);

   HYPRE_Real *v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
   HYPRE_Real *r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_fine.r));

   for (HYPRE_Int i = 0; i < fine_num_rows; i++){
      if (fabs(v_local_data[i] - r_local_data[i]) > 0.0) printf("residual failed: %e, %e\n", v_local_data[i], r_local_data[i]);
   }

   amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   A_array = hypre_ParAMGDataAArray(amg_data);
   HYPRE_Int gridk_first_row = hypre_ParCSRMatrixFirstRowIndex(A_array[0]);
   HYPRE_Int gridk_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));

   GridkSendRecv(dmem_all_data,
                 &(dmem_all_data->comm.gridk_r_inside_send),
                 r_local_data,
                 READ);
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
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.gridk_r_outside_recv),
                f_local_data,
                READ);
   hypre_MPI_Waitall(dmem_all_data->comm.gridk_r_inside_send.procs.size(),
                     dmem_all_data->comm.gridk_r_inside_send.requests,
                     MPI_STATUSES_IGNORE);
   hypre_MPI_Waitall(dmem_all_data->comm.gridk_r_outside_send.procs.size(),
                     dmem_all_data->comm.gridk_r_outside_send.requests,
                     MPI_STATUSES_IGNORE);

  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       for (HYPRE_Int i = 0; i < fine_num_rows; i++){
  //          printf("%d %d %e\n", my_id, fine_first_row+i, r_local_data[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       for (HYPRE_Int i = 0; i < gridk_num_rows; i++){
  //          printf("%d %d %e\n", my_id, gridk_first_row+i, f_local_data[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

