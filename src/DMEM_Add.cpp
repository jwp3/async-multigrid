#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Test.hpp"
#include "DMEM_Add.hpp"
#include "DMEM_Smooth.hpp"

void AddCycle(DMEM_AllData *dmem_all_data);
void AddResNorm(DMEM_AllData *dmem_all_data);
void PrintMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);
void PrintMessageFlags(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data);
void AsyncStart(DMEM_AllData *dmem_all_data);
void AsyncEnd(DMEM_AllData *dmem_all_data);
int CheckConverge(DMEM_AllData *dmem_all_data);
void DMEM_SyncAddResidual(DMEM_AllData *dmem_all_data);
void DMEM_SyncAddCorrect(DMEM_AllData *dmem_all_data);

void DMEM_Add(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   HYPRE_Int my_id_gridk, num_procs_gridk;
   MPI_Comm_rank(dmem_all_data->grid.my_comm, &my_id_gridk);
   MPI_Comm_size(dmem_all_data->grid.my_comm, &num_procs_gridk);

   HYPRE_Real vec_norm;
   hypre_Vector *vec;

   hypre_ParVector *r;

   dmem_all_data->iter.r_L2norm_local_converge_flag = 0;
   dmem_all_data->iter.grid_done_flag = 0;
   dmem_all_data->comm.all_done_flag = 0;
   dmem_all_data->comm.outside_done_flag = 0;

   int finest_level = 0;

   DMEM_ResetAllCommData(dmem_all_data);

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParAMGData *amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   MPI_Comm comm_gridk = dmem_all_data->grid.my_comm;
   hypre_ParVector *Vtemp_fine = hypre_ParAMGDataVtemp(amg_data_fine);
   hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);
   hypre_ParVector **F_array_fine = hypre_ParAMGDataFArray(amg_data_fine);
   hypre_ParVector **U_array_gridk = hypre_ParAMGDataUArray(amg_data_gridk);
   hypre_ParVector **U_array_fine = hypre_ParAMGDataUArray(amg_data_fine);
   hypre_ParCSRMatrix **A_array_gridk = hypre_ParAMGDataAArray(amg_data_gridk);
   hypre_ParCSRMatrix **A_array_fine = hypre_ParAMGDataAArray(amg_data_fine);
   HYPRE_Int num_rows_gridk = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_gridk[finest_level]));
   HYPRE_Int num_rows_fine = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array_fine[finest_level]));
 
   DMEM_HypreParVector_Set(dmem_all_data->vector_gridk.y, 0.0, num_rows_gridk);
   
   double start_begin = MPI_Wtime();

  // if (dmem_all_data->input.res_compute_type == LOCAL_RES){
     // DMEM_VectorToGridk_LocalRes(dmem_all_data, F_array_fine[finest_level], F_array_gridk[finest_level]);
      DMEM_VectorToGridk_LocalRes(dmem_all_data, dmem_all_data->vector_fine.r, F_array_gridk[finest_level]);
      if (dmem_all_data->input.async_flag == 1){
         DMEM_VectorToGridk_LocalRes(dmem_all_data, dmem_all_data->vector_fine.x, dmem_all_data->vector_gridk.x);
         DMEM_VectorToGridk_LocalRes(dmem_all_data, dmem_all_data->vector_fine.b, dmem_all_data->vector_gridk.b);
      }
  // }

   HYPRE_Real x0_norm_gridk_sum = 0, x0_norm_gridk;
   for (int i = 0; i < num_rows_gridk; i++) x0_norm_gridk_sum += hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x))[i];
   hypre_MPI_Allreduce(&x0_norm_gridk_sum, &x0_norm_gridk, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, dmem_all_data->grid.my_comm);

   if (dmem_all_data->input.async_flag == 1){
      AsyncStart(dmem_all_data);
   }
   dmem_all_data->output.start_wtime += MPI_Wtime() - start_begin;

   dmem_all_data->iter.cycle = 0;
   dmem_all_data->iter.relax = 0;

   int enter_add_loop_flag = 1;

   double solve_begin = MPI_Wtime();
   if (my_grid == 0){
      if (dmem_all_data->input.smoother == ASYNC_JACOBI ||
          dmem_all_data->input.smoother == ASYNC_L1_JACOBI ||
          dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL ||
          dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
          dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL){
         DMEM_HypreParVector_Copy(dmem_all_data->vector_gridk.r, F_array_gridk[finest_level], num_rows_gridk);
         DMEM_AsyncSmooth(dmem_all_data, finest_level);
        // if (dmem_all_data->input.async_flag == 0){
            enter_add_loop_flag = 0;
        // }      
      }
   }
   
   while (enter_add_loop_flag){
      if (dmem_all_data->input.async_flag == 1){
         dmem_all_data->iter.converge_flag = CheckConverge(dmem_all_data);
      }
      
      DMEM_DelayProc(dmem_all_data);
      AddCycle(dmem_all_data);

      if (dmem_all_data->input.async_flag == 1){
         DMEM_AddCorrect_LocalRes(dmem_all_data);
         DMEM_AddResidual_LocalRes(dmem_all_data);
      }
      else {
         DMEM_SyncAddCorrect(dmem_all_data);
         DMEM_SyncAddResidual(dmem_all_data);
      }

      if (dmem_all_data->comm.all_done_flag == 0){
         AddResNorm(dmem_all_data);
      }      

      if (dmem_all_data->input.async_flag == 0){
         dmem_all_data->iter.converge_flag = CheckConverge(dmem_all_data);
      }

      dmem_all_data->iter.cycle += 1;
      if (dmem_all_data->iter.converge_flag == 1){
         break;
      }
   }
   dmem_all_data->output.solve_wtime = MPI_Wtime() - solve_begin;

   double end_begin = MPI_Wtime();
   if (dmem_all_data->input.async_flag == 1){
      AsyncEnd(dmem_all_data);
   }
   dmem_all_data->output.end_wtime += MPI_Wtime() - end_begin;

   if (dmem_all_data->input.async_flag == 1){
      DMEM_SolutionToFinest_LocalRes(dmem_all_data, dmem_all_data->vector_gridk.x, dmem_all_data->vector_fine.x);
   }
   if (dmem_all_data->input.solver == MULTADD ||
       dmem_all_data->input.solver == BPX ||
       dmem_all_data->input.solver == AFACX){
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         dmem_all_data->matrix.A_fine,
                                         dmem_all_data->vector_fine.x,
                                         1.0,
                                         dmem_all_data->vector_fine.b,
                                         dmem_all_data->vector_fine.r);
      hypre_ParCSRMatrixMatvec(1.0,
                               dmem_all_data->matrix.A_fine,
                               dmem_all_data->vector_fine.x,
                               0.0,
                               dmem_all_data->vector_fine.e);
   }

//   HYPRE_Real x_norm_gridk = InnerProd(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x), hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x), dmem_all_data->grid.my_comm);
//   //HYPRE_Real x_norm_fine = InnerProd(hypre_ParVectorLocalVector(dmem_all_data->vector_fine.x), hypre_ParVectorLocalVector(dmem_all_data->vector_fine.x), MPI_COMM_WORLD);
//  // printf("%d %e %e\n", my_grid, x_norm_gridk, x_norm_fine);
//   
//   HYPRE_Real x_norm_gridk_sum = 0;
//   for (int i = 0; i < num_rows_gridk; i++) x_norm_gridk_sum += hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x))[i];
//   hypre_MPI_Allreduce(&x_norm_gridk_sum, &x_norm_gridk, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, dmem_all_data->grid.my_comm);
//   if (my_id_gridk == 0) printf("%d %.10e %.10e\n", my_grid, x0_norm_gridk, x_norm_gridk);
//   MPI_Barrier(MPI_COMM_WORLD);
//
  // for (HYPRE_Int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("id %d, grid %d:\n", my_id, my_grid);
  //       HYPRE_Real *x = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x));
  //       for (int i = 0; i < num_rows_gridk; i++){
  //          printf("\t%e\n", x[i]);
  //       }
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void AddCycle(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   HYPRE_Int num_rows;
   int fine_level, coarse_level;
   double smooth_begin, restrict_begin, prolong_begin, matvec_begin, vecop_begin, level_begin;
   double smooth_end,restrict_end, prolong_end, matvec_end, vecop_end, level_end;

   HYPRE_Int my_id_gridk, num_procs_gridk;
   MPI_Comm_rank(dmem_all_data->grid.my_comm, &my_id_gridk);
   MPI_Comm_size(dmem_all_data->grid.my_comm, &num_procs_gridk);

   HYPRE_Real vec_norm;
   hypre_Vector *vec;

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

   int finest_level = dmem_all_data->input.coarsest_mult_level;
   int coarsest_level;
   if (my_grid < num_levels-1 && dmem_all_data->input.solver == AFACX){
      coarsest_level = my_grid + 1;
   }
   else {
      coarsest_level = my_grid;
   }

   restrict_begin = matvec_begin = MPI_Wtime();
   if (dmem_all_data->input.num_interpolants == ONE_INTERPOLANT){
      if (my_grid > 0){
        // if (dmem_all_data->input.async_flag == 1){
        //    DMEM_AddCheckComm(dmem_all_data);
        // }
         hypre_ParCSRMatrixMatvecT(1.0,
                                   dmem_all_data->matrix.R_gridk,
                                   F_array[finest_level],
                                   0.0,
                                   F_array[coarsest_level]);
      }
   }
   else {
      for (int level = finest_level; level < coarsest_level; level++){
         level_begin = MPI_Wtime();
        // if (dmem_all_data->input.async_flag == 1){
        //    DMEM_AddCheckComm(dmem_all_data);
        // }
         fine_level = level;
         coarse_level = level + 1;
         hypre_ParCSRMatrixMatvecT(1.0,
                                   R_array[fine_level],
                                   F_array[fine_level],
                                   0.0,
                                   F_array[coarse_level]);
         dmem_all_data->output.level_wtime[level] += MPI_Wtime() - level_begin;
      }
   }
   matvec_end = restrict_end = MPI_Wtime();
   dmem_all_data->output.matvec_wtime += matvec_end - matvec_begin;
   dmem_all_data->output.restrict_wtime += restrict_end - restrict_begin;

  // if (dmem_all_data->input.async_flag == 1){
  //    DMEM_AddCheckComm(dmem_all_data);
  // }

   smooth_begin = MPI_Wtime();
   if (my_grid == dmem_all_data->grid.num_levels-1){
       hypre_GaussElimSolve(amg_data, coarsest_level, 9);
   }
   else {
      DMEM_AddSmooth(dmem_all_data, coarsest_level);
   }
   dmem_all_data->output.smooth_wtime += MPI_Wtime() - smooth_begin;
  
  // if (dmem_all_data->input.async_flag == 1){
  //    DMEM_AddCheckComm(dmem_all_data);
  // }
   prolong_begin = matvec_begin = MPI_Wtime();
   if (dmem_all_data->input.num_interpolants == ONE_INTERPOLANT){
      if (my_grid > 0){
         double level_begin, comp_begin;
         hypre_ParCSRMatrixMatvec(1.0,
                                  dmem_all_data->matrix.P_gridk,
                                  U_array[coarsest_level],
                                  0.0,
                                  U_array[finest_level]);
        // hypre_ParCSRMatrixMatvecOutOfPlace(1.0,
        //                                    dmem_all_data->matrix.P_gridk,
        //                                    U_array[coarsest_level],
        //                                    0.0,
        //                                    Vtemp,
        //                                    U_array[finest_level]);
      }
     // if (dmem_all_data->input.async_flag == 1){
     //    DMEM_AddCheckComm(dmem_all_data);
     // }
   }
   else {
      for (int level = coarsest_level; level > finest_level; level--){
         level_begin = MPI_Wtime();
         fine_level = level-1;
         coarse_level = level;
        // hypre_ParCSRMatrixMatvec(1.0,
        //                          P_array[fine_level], 
        //                          U_array[coarse_level],
        //                          0.0,
        //                          U_array[fine_level]);
         hypre_ParCSRMatrixMatvecOutOfPlace(1.0,
                                            P_array[fine_level],
                                            U_array[coarse_level],
                                            0.0,
                                            Vtemp,
                                            U_array[fine_level]);
        // if (dmem_all_data->input.async_flag == 1){
        //    DMEM_AddCheckComm(dmem_all_data);
        // }
         dmem_all_data->output.level_wtime[level] += MPI_Wtime() - level_begin;
      }
   }
   matvec_end = prolong_end = MPI_Wtime();
   dmem_all_data->output.matvec_wtime += matvec_end - matvec_begin;
   dmem_all_data->output.prolong_wtime += prolong_end - prolong_begin;

   num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
   if (dmem_all_data->input.async_flag == 1){
      if (dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL){
         DMEM_ChebyUpdate(dmem_all_data, dmem_all_data->vector_gridk.d, U_array[0], num_rows);
      }
   }

  // if (dmem_all_data->input.async_flag == 1){
  //    DMEM_AddCheckComm(dmem_all_data);
  // }
}

void AddResNorm(DMEM_AllData *dmem_all_data)
{
   double residual_norm_begin;
   int finest_level = 0;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   hypre_ParAMGData *amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   MPI_Comm comm_gridk = dmem_all_data->grid.my_comm;
   HYPRE_Int num_procs_gridk;
   MPI_Comm_size(comm_gridk, &num_procs_gridk);

   if (dmem_all_data->input.async_flag == 1){
      hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);
      HYPRE_Real my_comm_done;
      hypre_Vector *f = hypre_ParVectorLocalVector(F_array_gridk[finest_level]);
      residual_norm_begin = MPI_Wtime();
      if (dmem_all_data->input.async_type == SEMI_ASYNC){
        // HYPRE_Real *z = (HYPRE_Real *)malloc((dmem_all_data->grid.num_levels+1) * sizeof(HYPRE_Real));
        // z[0] = (HYPRE_Real)(dmem_all_data->comm.outside_done_flag);
        // // TODO: stopped here, need grid numbers for neighbs
        // for (int i = 0; i < dmem_all_data->grid.num_levels; i++){
        //    z[i+1] = ;
        // }
        // HYPRE_Real *v = InnerProdFlagV(f, f, comm_gridk, z, &my_comm_done);
        // dmem_all_data->iter.r_L2norm_local = sqrt(v[0])/dmem_all_data->output.r0_norm2;
        // dmem_all_data->comm.outside_done_flag = v[1];
        // for (int i = 0; i < dmem_all_data->grid.num_levels; i++){
        //    
        // }
        // free(v);
      }
      else {
         dmem_all_data->iter.r_L2norm_local = sqrt(InnerProdFlag(f, f, comm_gridk, (HYPRE_Real)(dmem_all_data->comm.outside_done_flag), &my_comm_done))/dmem_all_data->output.r0_norm2;
      }
      dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
      if (dmem_all_data->iter.r_L2norm_local < dmem_all_data->input.tol){
         dmem_all_data->iter.r_L2norm_local_converge_flag = 1;
      }
      if ((HYPRE_Int)my_comm_done == num_procs_gridk){
         dmem_all_data->comm.all_done_flag = 1;
      }
   }
   else {
      if (dmem_all_data->iter.cycle % dmem_all_data->input.async_comm_save_divisor == 0){
         hypre_ParVector *r = dmem_all_data->vector_fine.r;
         residual_norm_begin = MPI_Wtime();
         HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r));
         dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
         if (res_norm/dmem_all_data->output.r0_norm2 < dmem_all_data->input.tol){
            dmem_all_data->iter.r_L2norm_local_converge_flag = 1;
         }
      }
   }
 
  // if (dmem_all_data->input.async_flag == 1){
  //    DMEM_AddCheckComm(dmem_all_data);
  // }
}

void DMEM_AddCorrect_LocalRes(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   double vecop_begin, comm_begin, mpiwait_begin;
   double vecop_end, comm_end, mpiwait_end;
   int recv_flag;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector **U_array;
   hypre_ParVector *e, *x, *y, *d;
   HYPRE_Real *e_local_data, *x_local_data, *y_local_data, *u_local_data;
   HYPRE_Int num_rows;

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   A_array = hypre_ParAMGDataAArray(amg_data);
   num_rows = hypre_ParCSRMatrixNumRows(A_array[finest_level]);

   U_array = hypre_ParAMGDataUArray(amg_data);
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[finest_level]));

   x = dmem_all_data->vector_gridk.x;
   y = dmem_all_data->vector_gridk.y;
   d = dmem_all_data->vector_gridk.d;
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   y_local_data = hypre_VectorData(hypre_ParVectorLocalVector(y));

  // int first_cheby_iter_flag = 0;
  // if (dmem_all_data->iter.cycle == 0 && (dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL)){
  //    first_cheby_iter_flag = 1;
  // }

   vecop_begin = MPI_Wtime();
   DMEM_HypreParVector_Axpy(y, U_array[0], 1.0, num_rows);
   dmem_all_data->output.vecop_wtime = MPI_Wtime() - vecop_begin;
   if (dmem_all_data->iter.converge_flag == 1 ||
       dmem_all_data->iter.cycle % dmem_all_data->input.async_comm_save_divisor == 0){
      comm_begin = MPI_Wtime();
     // if (dmem_all_data->input.async_flag == 0/* || first_cheby_iter_flag == 1*/){
     //    hypre_MPI_Waitall(dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(),
     //                      dmem_all_data->comm.gridjToGridk_Correct_outsideSend.requests,
     //                      MPI_STATUSES_IGNORE);
     // }
      SendRecv(dmem_all_data,
               &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend),
               y_local_data,
               ACCUMULATE);
      dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;

      vecop_begin = MPI_Wtime();
      DMEM_HypreParVector_Set(y, 0.0, num_rows);
      dmem_all_data->output.vecop_wtime = MPI_Wtime() - vecop_begin;
   }

   vecop_begin = MPI_Wtime();
   DMEM_HypreParVector_Axpy(x, U_array[0], 1.0, num_rows);
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

  // if (dmem_all_data->input.async_flag == 1){
      DMEM_AddCheckComm(dmem_all_data);
  // }
}

void DMEM_AddCheckComm(DMEM_AllData *dmem_all_data)
{
   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector *e, *x, *d;
   HYPRE_Real *e_local_data, *x_local_data;
   int flag;
   double comm_begin, vecop_begin;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   int recv_flag;
   amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   A_array = hypre_ParAMGDataAArray(amg_data);
                    
   e = dmem_all_data->vector_gridk.e;
   x = dmem_all_data->vector_gridk.x;
   d = dmem_all_data->vector_gridk.d;
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[finest_level]);

  // int first_cheby_iter_flag = 0;
  // if (dmem_all_data->iter.cycle == 0 && (dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL)){
  //    first_cheby_iter_flag = 1;
  // }
   
   vecop_begin = MPI_Wtime();
   DMEM_HypreParVector_Set(e, 0.0, num_rows);
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

   recv_flag = 0;
   comm_begin = MPI_Wtime();
  // if (dmem_all_data->input.async_flag == 0 || first_cheby_iter_flag == 0){
      recv_flag = SendRecv(dmem_all_data,
                           &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                           e_local_data,
                           ACCUMULATE);
  // }
  // if (dmem_all_data->input.async_flag == 0/* || first_cheby_iter_flag == 1*/){
  //    DMEM_HypreParVector_Set(e, 0.0, num_rows);
  //    CompleteRecv(dmem_all_data,
  //                 &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
  //                 e_local_data,
  //                 ACCUMULATE);
  //   // if (dmem_all_data->input.async_flag == 1){
  //   //    DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv));
  //   // }
  // }
   dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;

   if (recv_flag == 1/* || first_cheby_iter_flag == 1*/){
      vecop_begin = MPI_Wtime();
      DMEM_HypreParVector_Axpy(x, e, 1.0, num_rows);
      if ((dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL) &&
          dmem_all_data->grid.my_grid == dmem_all_data->input.cheby_grid){
         DMEM_HypreParVector_Axpy(d, e, 1.0, num_rows);
      }
      dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
   }

   if (dmem_all_data->input.async_flag == 1){
      comm_begin = MPI_Wtime();
      for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(); i++){
         CheckInFlight(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), i);
      }
      dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
   }
}

void DMEM_AddResidual_LocalRes(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int my_grid = dmem_all_data->grid.my_grid;
   double begin;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   hypre_ParAMGData *amg_data_gridk =
      (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);
   hypre_ParCSRMatrix **A_array_gridk = hypre_ParAMGDataAArray(amg_data_gridk);

   double residual_begin = MPI_Wtime();
   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                      A_array_gridk[finest_level],
                                      dmem_all_data->vector_gridk.x,
                                      1.0,
                                      dmem_all_data->vector_gridk.b,
                                      F_array_gridk[finest_level]);
   dmem_all_data->output.residual_wtime += MPI_Wtime() - residual_begin;
  // if (dmem_all_data->input.async_flag == 1){
  //    DMEM_AddCheckComm(dmem_all_data);
  // }
}

void DMEM_SolutionToFinest_LocalRes(DMEM_AllData *dmem_all_data,
                                    hypre_ParVector *x_gridk,
                                    hypre_ParVector *x_fine)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   int my_grid = dmem_all_data->grid.my_grid;
   double begin;

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   HYPRE_Real *x_gridk_local_data, *x_fine_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   x_gridk_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x_gridk));
   x_fine_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x_fine));

  // double comm_begin = MPI_Wtime();
   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Correct_insideSend.requests,
                     MPI_STATUSES_IGNORE);
  // dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - comm_begin;

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Correct_insideSend),
            x_gridk_local_data,
            WRITE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
            NULL,
            READ);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                x_fine_local_data,
                READ);
  // dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
}

void DMEM_VectorToGridk_LocalRes(DMEM_AllData *dmem_all_data,
                                 hypre_ParVector *r,
                                 hypre_ParVector *f)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   int my_grid = dmem_all_data->grid.my_grid;
   double comm_begin, mpiwait_begin;

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   HYPRE_Real *r_local_data, *f_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(r));
   f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(f));

   if (dmem_all_data->input.async_flag == 0){
      comm_begin = mpiwait_begin = MPI_Wtime();
   }
   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Residual_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   if (dmem_all_data->input.async_flag == 0){
      dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - mpiwait_begin;
   }
   

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Residual_insideSend),
            r_local_data,
            WRITE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
            NULL,
            READ);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Residual_insideRecv),
                f_local_data,
                READ);
   if (dmem_all_data->input.async_flag == 0){
      dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
   }
}

void DMEM_SyncAddCorrect(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   int finest_level = dmem_all_data->input.coarsest_mult_level;
   double comm_begin, mpiwait_begin, vecop_begin;

   hypre_ParAMGData *fine_amg_data, *gridk_amg_data;
   hypre_ParCSRMatrix **A_array_fine, **A_array_gridk;
   hypre_ParVector **U_array;
   hypre_ParVector *e, *x, *d;
   HYPRE_Real *e_local_data, *x_local_data, *u_local_data;
   HYPRE_Int fine_num_rows, gridk_num_rows;


   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;

   fine_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver;
   gridk_amg_data = (hypre_ParAMGData*)dmem_all_data->hypre.solver_gridk;

   A_array_fine = hypre_ParAMGDataAArray(fine_amg_data);
   A_array_gridk = hypre_ParAMGDataAArray(gridk_amg_data);

   U_array = hypre_ParAMGDataUArray(gridk_amg_data);
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));

   fine_num_rows = hypre_ParCSRMatrixNumRows(dmem_all_data->matrix.A_fine);

   e = dmem_all_data->vector_fine.e;
   x = dmem_all_data->vector_fine.x;
   d = dmem_all_data->vector_fine.d;
   vecop_begin = MPI_Wtime();
   DMEM_HypreParVector_Set(e, 0.0, fine_num_rows);
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));


   comm_begin = mpiwait_begin = MPI_Wtime();
   hypre_MPI_Waitall(dmem_all_data->comm.finestToGridk_Correct_insideSend.procs.size(),
                     dmem_all_data->comm.finestToGridk_Correct_insideSend.requests,
                     MPI_STATUSES_IGNORE);
   dmem_all_data->output.mpiwait_wtime += MPI_Wtime() - mpiwait_begin;

   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Correct_insideSend),
            u_local_data,
            ACCUMULATE);
   SendRecv(dmem_all_data,
            &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
            NULL,
            ACCUMULATE);
   CompleteRecv(dmem_all_data,
                &(dmem_all_data->comm.finestToGridk_Correct_insideRecv),
                e_local_data,
                ACCUMULATE);
   dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;

   vecop_begin = MPI_Wtime();
   if (dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL){
      DMEM_ChebyUpdate(dmem_all_data, d, e, fine_num_rows);
      DMEM_HypreParVector_Axpy(x, d, 1.0, fine_num_rows);
   }
   else {
      DMEM_HypreParVector_Axpy(x, e, 1.0, fine_num_rows);
   }
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
}

void DMEM_SyncAddResidual(DMEM_AllData *dmem_all_data)
{
   double begin;
   hypre_ParVector *r = dmem_all_data->vector_fine.r;
   
   double residual_begin = MPI_Wtime();
   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                      dmem_all_data->matrix.A_fine,
                                      dmem_all_data->vector_fine.x,
                                      1.0,
                                      dmem_all_data->vector_fine.b,
                                      dmem_all_data->vector_fine.r);
   dmem_all_data->output.residual_wtime += MPI_Wtime() - residual_begin;

   hypre_ParAMGData *amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);

   DMEM_VectorToGridk_LocalRes(dmem_all_data, dmem_all_data->vector_fine.r, F_array_gridk[0]);
}

void PrintMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   for (HYPRE_Int i = 0; i < comm_data->procs.size(); i++){
//      if (comm_data->message_count[i] < dmem_all_data->input.num_inner_cycles)
      if (comm_data->type == GRIDK_OUTSIDE_SEND || comm_data->type == FINE_INTRA_OUTSIDE_SEND){
         printf("send ");
      }
      else if (comm_data->type == GRIDK_OUTSIDE_RECV || comm_data->type == FINE_INTRA_OUTSIDE_RECV){
         printf("recv ");
      }
      printf("%d %d %d\n", my_id, comm_data->procs[i], comm_data->message_count[i]);
   }
}

int DMEM_CheckMessageCount(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   for (int i = 0; i < comm_data->procs.size(); i++){
      if (comm_data->message_count[i] < dmem_all_data->input.num_inner_cycles){
         return 0;
      }
   }
   return 1;
}

void PrintMessageFlags(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   for (int i = 0; i < comm_data->procs.size(); i++){
      if (comm_data->done_flags[i] != 2) printf("bad flag %d\n", comm_data->done_flags[i]);
   }
}

int DMEM_CheckMessageFlagsValue(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int value)
{
   for (int i = 0; i < comm_data->procs.size(); i++){
      if (comm_data->done_flags[i] != value){
         return 0;
      }
   }
   return 1;
}

int DMEM_CheckMessageFlagsNotValue(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data, int value)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   for (int i = 0; i < comm_data->procs.size(); i++){
      if (comm_data->done_flags[i] == value){
         return 0;
      }
   }
   return 1;
}

int DMEM_CheckOutsideDoneFlag(DMEM_AllData *dmem_all_data)
{
   if (DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), 0) == 1 &&
       DMEM_CheckMessageFlagsNotValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv), 0) == 1){
      dmem_all_data->comm.outside_done_flag = 1;
      return 1;
   }
   return 0;
}

void DMEM_AsyncRecvStart(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   MPI_Status status;
   int flag;
   for (int i = 0; i < comm_data->procs.size(); i++){
      HYPRE_Int ip = comm_data->procs[i];
      HYPRE_Int vec_len = comm_data->len[i];
      hypre_MPI_Irecv(comm_data->data[i],
                      vec_len+2,
                      HYPRE_MPI_REAL,
                      ip,
                      comm_data->tag,
                      MPI_COMM_WORLD,
                      &(comm_data->requests[i]));
   }
}

void AsyncStart(DMEM_AllData *dmem_all_data)
{
   DMEM_AsyncRecvStart(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv));
}

void AsyncRecvCleanup(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParVector *e, *x, *y;
   HYPRE_Real *e_local_data, *x_local_data, *y_local_data;
   HYPRE_Int num_rows;
   double vecop_begin, comm_begin;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   e = dmem_all_data->vector_gridk.e;
   x = dmem_all_data->vector_gridk.x;
   y = dmem_all_data->vector_gridk.y;
   amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   A_array = hypre_ParAMGDataAArray(amg_data);
   num_rows = hypre_ParCSRMatrixNumRows(A_array[finest_level]);

   e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   y_local_data = hypre_VectorData(hypre_ParVectorLocalVector(y));

  // vecop_begin = MPI_Wtime();
   DMEM_HypreParVector_Set(e, 0.0, num_rows);
  // dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

   while (1){
      if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
         if (DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv), 2) == 1 &&
DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend), 2) == 1){
            break;
         }
      }
      else {
         if (DMEM_CheckMessageFlagsValue(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv), 2) == 1){
            break;
         }
      }
      int recv_flag;
     // comm_begin = MPI_Wtime();
      recv_flag = SendRecv(dmem_all_data,
                           &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv),
                           e_local_data,
                           ACCUMULATE);
      if (dmem_all_data->input.converge_test_type == LOCAL_CONVERGE){
         DMEM_HypreParVector_Set(y, 0.0, num_rows);
         SendRecv(dmem_all_data,
                  &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend),
                  y_local_data,
                  ACCUMULATE);
      }
     // dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
   }
   
   vecop_begin = MPI_Wtime();
   DMEM_HypreParVector_Axpy(x, e, 1.0, num_rows);
   dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

  // comm_begin = MPI_Wtime();
   CompleteInFlight(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend));
  // dmem_all_data->output.comm_wtime += MPI_Wtime() - comm_begin;
}

void AsyncEnd(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->grid.my_grid == 0){
      if (dmem_all_data->input.smoother == ASYNC_JACOBI ||
          dmem_all_data->input.smoother == ASYNC_L1_JACOBI ||
          dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL ||
          dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
          dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL){
         DMEM_AsyncSmoothEnd(dmem_all_data);
      }
   }
   AsyncRecvCleanup(dmem_all_data);
}

int CheckConverge(DMEM_AllData *dmem_all_data)
{
   double comm_begin;
   int cycle, num_cycles;

   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   if (dmem_all_data->input.async_flag == 1 && dmem_all_data->input.converge_test_type == GLOBAL_CONVERGE){
      /* check for local convergence and comm convergence */
      if (dmem_all_data->comm.all_done_flag == 0){
         /* check local convergence */
         if (dmem_all_data->iter.grid_done_flag == 0){
            if (dmem_all_data->iter.cycle >= dmem_all_data->input.num_cycles-1 ||
                dmem_all_data->iter.r_L2norm_local_converge_flag == 1){
               dmem_all_data->iter.grid_done_flag = 1;
            }
         }
         /* check comm convergence */
         if (dmem_all_data->iter.grid_done_flag == 1 && dmem_all_data->comm.outside_done_flag == 0){
            DMEM_CheckOutsideDoneFlag(dmem_all_data);
         }
         return 0;
      }
      else {
         return 1;
      }
   }
   else {
      if (dmem_all_data->iter.cycle >= dmem_all_data->input.num_cycles-1 ||
          dmem_all_data->iter.r_L2norm_local_converge_flag == 1){
         dmem_all_data->iter.grid_done_flag = 1;
         return 1;
      }
      else {
         return 0;
      }
   }
}
