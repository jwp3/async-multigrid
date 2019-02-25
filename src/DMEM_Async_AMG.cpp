#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"

void DMEM_FineResidual(DMEM_AllData *dmem_all_data);

void DMEM_Async_AMG(DMEM_AllData *dmem_all_data)
{
   DMEM_FineResidual(dmem_all_data);
}

void DMEM_FineResidual(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData*)dmem_all_data->hypre.solver;

   hypre_ParCSRMatrix *A = dmem_all_data->matrix.A;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);

   hypre_ParVector *x = dmem_all_data->vector.x;
   hypre_ParVector *b = dmem_all_data->vector.b;
   hypre_ParVector *r = dmem_all_data->vector.r;
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
  // CompleteRecv(dmem_all_data,
  //              &(dmem_all_data->comm.fine_outside_recv),
  //              hypre_VectorData(x_ghost));

   hypre_CSRMatrixMatvec(-1.0,
                         offd,
                         x_ghost,
                         1.0,
                         r_local);
   MPI_Waitall(dmem_all_data->comm.fine_inside_send.procs.size(),
               dmem_all_data->comm.fine_inside_send.requests,
               MPI_STATUSES_IGNORE); 
  // MPI_Waitall(dmem_all_data->comm.fine_outside_send.procs.size(),
  //             dmem_all_data->comm.fine_outside_send.requests,
  //             MPI_STATUSES_IGNORE);

  // hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
  // hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
  //                                    A,
  //                                    x,
  //                                    1.0,
  //      		              b,
  //                                    Vtemp);

  // 

  // HYPRE_Real *r_local_data = hypre_VectorData(r_local);
  // HYPRE_Real *x_local_data = hypre_VectorData(x_local);
  // HYPRE_Real *v = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
  // for (int p = 0; p < num_procs; p++){ 
  //    if (my_id == p)
  //    for (int i = 0; i < hypre_VectorSize(r_local); i++){
  //       if (fabs(r_local_data[i] - v[i]) > 0) printf("%e\n", r_local_data[i] - v[i]);
  //      // printf("%e, %e, %e\n", r_local_data[i], x_local_data[i], v[i]);
  //      // v[i] = 0;
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
 
  // amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_local;
  // hypre_Vector *w_local = hypre_ParVectorLocalVector(hypre_ParAMGDataVtemp(amg_data));
  // HYPRE_Real *w_local_data = hypre_VectorData(w_local); 
  // printf("%d: %d %d %d %d\n",
  //        my_id,
  //        dmem_all_data->comm.gridk_r_outside_recv.start[0],
  //        dmem_all_data->comm.gridk_r_outside_recv.end[0],
  //        dmem_all_data->comm.gridk_r_outside_recv.len[0],
  //        hypre_VectorSize(w_local));

  // GridkSendRecv(dmem_all_data,
  //               &(dmem_all_data->comm.gridk_r_inside_send),
  //               r_local_data);
  // GridkSendRecv(dmem_all_data,
  //               &(dmem_all_data->comm.gridk_r_outside_send),
  //               r_local_data);
  // GridkSendRecv(dmem_all_data,
  //               &(dmem_all_data->comm.gridk_r_inside_recv),
  //               NULL);
  // GridkSendRecv(dmem_all_data,
  //               &(dmem_all_data->comm.gridk_r_outside_recv),
  //               w_local_data); 
  // CompleteRecv(dmem_all_data,
  //              &(dmem_all_data->comm.gridk_r_inside_recv),
  //              w_local_data);
  // CompleteRecv(dmem_all_data,
  //              &(dmem_all_data->comm.gridk_r_outside_recv),
  //              w_local_data);
  // MPI_Waitall(dmem_all_data->comm.gridk_r_inside_send.procs.size(),
  //             dmem_all_data->comm.gridk_r_inside_send.requests,
  //             MPI_STATUSES_IGNORE);
  // MPI_Waitall(dmem_all_data->comm.gridk_r_outside_send.procs.size(),
  //             dmem_all_data->comm.gridk_r_outside_send.requests,
  //             MPI_STATUSES_IGNORE);
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p)
  //    for (int i = 0; i < hypre_VectorSize(w_local); i++){
  //       printf("%e\n", w_local_data[i]);
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }

//   GridkSendRecv(dmem_all_data,
//                 &(dmem_all_data->comm.gridk_e_inside_send),
//                 w_local_data);
//   GridkSendRecv(dmem_all_data,
//                 &(dmem_all_data->comm.gridk_e_outside_send),
//                 w_local_data);
//   GridkSendRecv(dmem_all_data,
//                 &(dmem_all_data->comm.gridk_e_inside_recv),
//                 NULL);
//   GridkSendRecv(dmem_all_data,
//                 &(dmem_all_data->comm.gridk_e_outside_recv),
//                 x_local_data);
//
//   CompleteRecv(dmem_all_data,
//                &(dmem_all_data->comm.gridk_e_inside_recv),
//                x_local_data);
//   CompleteRecv(dmem_all_data,
//                &(dmem_all_data->comm.gridk_e_outside_recv),
//                x_local_data);
//  // printf("%d\n", my_id);
//   MPI_Waitall(dmem_all_data->comm.gridk_e_inside_send.procs.size(),
//               dmem_all_data->comm.gridk_e_inside_send.requests,
//               MPI_STATUSES_IGNORE);
//   MPI_Waitall(dmem_all_data->comm.gridk_e_outside_send.procs.size(),
//               dmem_all_data->comm.gridk_e_outside_send.requests,
//               MPI_STATUSES_IGNORE);
//
//   for (int p = 0; p < num_procs; p++){
//      if (my_id == p)
//      for (int i = 0; i < hypre_VectorSize(r_local); i++){
//         if (fabs(x_local_data[i] - v[i]) > 0) printf("%e, %e\n", x_local_data[i], v[i]);
//        // printf("%e, %e, %e\n", x_local_data[i], r_local_data[i], v[i]);
//      }
//      MPI_Barrier(MPI_COMM_WORLD);
//   }
}
