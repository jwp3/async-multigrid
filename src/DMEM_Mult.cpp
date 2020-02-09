#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Add.hpp"
#include "_hypre_utilities.h"
#include "DMEM_Misc.hpp"
#include "DMEM_Mult.hpp"

void DMEM_Mult(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->input.num_cycles <= 0) return;

   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   double vecop_begin, matvec_begin, residual_norm_begin;
   double vecop_end, matvec_end, residual_norm_end;

   hypre_ParAMGData *amg_data =
      (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[0]); 

   HYPRE_Real *y_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_fine.y));
   HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
   
   DMEM_HypreParVector_Copy(U_array[0], dmem_all_data->vector_fine.x, num_rows);
   DMEM_HypreParVector_Copy(F_array[0], dmem_all_data->vector_fine.b, num_rows);
  
   dmem_all_data->iter.cycle = 0;

   double mu, delta;
   if (dmem_all_data->input.accel_type == CHEBY_ACCEL){
      mu = dmem_all_data->cheby.mu;
      delta = dmem_all_data->cheby.delta;
   }
   
   double solve_begin = MPI_Wtime();
   while (1){
      if (dmem_all_data->input.accel_type == CHEBY_ACCEL){
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Set(U_array[0], 0.0, num_rows);
         DMEM_HypreParVector_Copy(F_array[0], dmem_all_data->vector_fine.r, num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
      }

      DMEM_MultCycle(dmem_all_data);

      if (dmem_all_data->input.accel_type == CHEBY_ACCEL){
         if (dmem_all_data->iter.cycle == 0){
            vecop_begin = MPI_Wtime();
            DMEM_HypreParVector_Copy(dmem_all_data->vector_fine.y, U_array[0], num_rows);
            dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
         }
         else {
            double c_temp = dmem_all_data->cheby.c;
            dmem_all_data->cheby.c = 2.0 * mu * dmem_all_data->cheby.c - dmem_all_data->cheby.c_prev;
            dmem_all_data->cheby.c_prev = c_temp;
            double omega = 2.0 * mu * dmem_all_data->cheby.c_prev / dmem_all_data->cheby.c;

            vecop_begin = MPI_Wtime();
           // DMEM_HypreParVector_Scale(dmem_all_data->vector_fine.y, 1.0-omega, num_rows);
           // DMEM_HypreParVector_Axpy(dmem_all_data->vector_fine.y, U_array[0], omega * delta, num_rows);
            for (int i = 0; i < num_rows; i++){
               y_local_data[i] = (omega - 1.0) * y_local_data[i] + omega * delta * u_local_data[i];
            }
            dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
         }
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Axpy(dmem_all_data->vector_fine.x, dmem_all_data->vector_fine.y, 1.0, num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
      }
      else { 
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Copy(dmem_all_data->vector_fine.x, U_array[0], num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
      }

      matvec_begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[0],
                                         dmem_all_data->vector_fine.x,
                                         1.0,
                                         dmem_all_data->vector_fine.b,
                                         dmem_all_data->vector_fine.r);
      matvec_end = MPI_Wtime();
      dmem_all_data->output.residual_wtime += matvec_end - matvec_begin;
      dmem_all_data->output.matvec_wtime += matvec_end - matvec_begin;

      hypre_ParVector *r = dmem_all_data->vector_fine.r;
      residual_norm_begin = MPI_Wtime();
      HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r));
      dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
      dmem_all_data->iter.cycle += 1;
      if (res_norm/dmem_all_data->output.r0_norm2 < dmem_all_data->input.tol || dmem_all_data->iter.cycle == dmem_all_data->input.num_cycles) break;
   }
   dmem_all_data->output.solve_wtime = MPI_Wtime() - solve_begin;
   hypre_ParCSRMatrixMatvecOutOfPlace(1.0,
                                      dmem_all_data->matrix.A_fine,
                                      dmem_all_data->vector_fine.x,
                                      0.0,
                                      Vtemp,
                                      dmem_all_data->vector_fine.e);
}

void DMEM_MultCycle(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   double smooth_begin, restrict_begin, prolong_begin, matvec_begin, vecop_begin, level_begin;
   double smooth_end, restrict_end, prolong_end, matvec_end, vecop_end, level_end;

   HYPRE_Real *u_local_data;
   HYPRE_Real *v_local_data;
   HYPRE_Real *f_local_data;
   HYPRE_Real *r_local_data;
   HYPRE_Real *x_local_data;
   HYPRE_Real *A_data;
   HYPRE_Int *A_i;
   HYPRE_Int num_rows;

   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
   hypre_ParVector *Ztemp = hypre_ParAMGDataZtemp(amg_data);

   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   int coarsest_level = dmem_all_data->grid.num_levels-1;
   if (dmem_all_data->input.solver == MULT_MULTADD){
      coarsest_level = dmem_all_data->input.coarsest_mult_level;
   }

   for (HYPRE_Int level = 0; level < coarsest_level; level++){
      level_begin = MPI_Wtime();
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;

      hypre_ParCSRMatrix *A = A_array[fine_grid];

      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      num_rows = hypre_ParCSRMatrixNumRows(A);
      A_data = hypre_CSRMatrixData(A_diag);
      A_i = hypre_CSRMatrixI(A_diag);

      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid]));
      r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_fine.r));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
  
      /* smooth */
      smooth_begin = vecop_begin = MPI_Wtime();
      if (level == 0){
         DMEM_HypreParVector_Copy(Vtemp, dmem_all_data->vector_fine.r, num_rows);
      }
      else {
         DMEM_HypreParVector_Copy(Vtemp, F_array[fine_grid], num_rows);
         DMEM_HypreParVector_Set(U_array[fine_grid], 0.0, num_rows);
      }
      dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

      if (dmem_all_data->input.smoother == L1_JACOBI){
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, dmem_all_data->matrix.L1_row_norm_fine[fine_grid], num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
      }
      else if (dmem_all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL ||
               dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL){
         hypre_BoomerAMGRelax(A,
                              F_array[fine_grid],
                              NULL,
                              3,
                              0,
                              1.0,
                              1.0,
                              NULL,
                              U_array[fine_grid], 
                              dmem_all_data->vector_fine.e,
                              Ztemp);
      }
      else {
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, dmem_all_data->matrix.wJacobi_scale_fine[fine_grid], num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
      }

      vecop_begin = MPI_Wtime(); 
      DMEM_HypreParVector_Copy(dmem_all_data->vector_fine.e, F_array[fine_grid], num_rows);
      dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

      matvec_begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[fine_grid],
                                         U_array[fine_grid],
                                         1.0,
                                         dmem_all_data->vector_fine.e,
                                         Vtemp);
      dmem_all_data->output.smooth_wtime += MPI_Wtime() - smooth_begin;
      /* restrict */
      restrict_begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
        			Vtemp,
                                0.0,
                                F_array[coarse_grid]);
      level_end = MPI_Wtime();
      dmem_all_data->output.matvec_wtime += level_end - matvec_begin;
      dmem_all_data->output.restrict_wtime += level_end - restrict_begin;
      dmem_all_data->output.level_wtime[level] += level_end - level_begin;
   }

   double coarse_r0_norm2, res_norm;
   
   smooth_begin = MPI_Wtime();
   hypre_GaussElimSolve(amg_data, coarsest_level, 9);
   dmem_all_data->output.smooth_wtime += MPI_Wtime() - smooth_begin;
    
   for (HYPRE_Int level = coarsest_level; level > 0; level--){
      level_begin = MPI_Wtime();
      HYPRE_Int fine_grid = level - 1;
      HYPRE_Int coarse_grid = level;

      vecop_begin = MPI_Wtime();
      DMEM_HypreParVector_Copy(dmem_all_data->vector_fine.e, U_array[fine_grid], num_rows);
      dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

      /* prolong and correct */
      prolong_begin = matvec_begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvecOutOfPlace(1.0,
                                         P_array[fine_grid], 
                                         U_array[coarse_grid],
                                         1.0,
                                         dmem_all_data->vector_fine.e,
                                         U_array[fine_grid]);
      prolong_end = matvec_end = MPI_Wtime();
      dmem_all_data->output.matvec_wtime += matvec_end - matvec_begin;
      dmem_all_data->output.prolong_wtime += prolong_end - prolong_begin;
      
      /* smooth */
      hypre_ParCSRMatrix *A = A_array[fine_grid];
      A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));

      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid]));
     
      vecop_begin = MPI_Wtime(); 
      DMEM_HypreParVector_Copy(dmem_all_data->vector_fine.e, F_array[fine_grid], num_rows);
      dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

      /* smooth */
      matvec_begin = smooth_begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[fine_grid],
                                         U_array[fine_grid],
                                         1.0,
                                         dmem_all_data->vector_fine.e,
                                         Vtemp);
      dmem_all_data->output.matvec_wtime += MPI_Wtime() - matvec_begin;

      if (dmem_all_data->input.smoother == L1_JACOBI){
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, dmem_all_data->matrix.L1_row_norm_fine[fine_grid], num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
      }
      else if (dmem_all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL ||
               dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL){
         hypre_BoomerAMGRelax(A,
                              F_array[fine_grid],
                              NULL,
                              4,
                              0,
                              1.0,
                              1.0,
                              NULL,
                              U_array[fine_grid],
                              dmem_all_data->vector_fine.e,
                              Ztemp);
      }
      else {
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, dmem_all_data->matrix.wJacobi_scale_fine[fine_grid], num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
      }
      dmem_all_data->output.smooth_wtime += MPI_Wtime() - smooth_begin;
      dmem_all_data->output.level_wtime[level] += MPI_Wtime() - level_begin;
   }
}

void DMEM_SyncAdd(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);

   dmem_all_data->iter.cycle = 0;

   double begin = MPI_Wtime();
   while(1){
      DMEM_SyncAddCycle(dmem_all_data, dmem_all_data->hypre.solver, dmem_all_data->iter.cycle);
      double residual_begin = MPI_Wtime();
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[0],
                                         dmem_all_data->vector_fine.x,
                                         1.0,
                                         dmem_all_data->vector_fine.b,
                                         dmem_all_data->vector_fine.r);
      dmem_all_data->output.residual_wtime += MPI_Wtime() - residual_begin;
      double residual_norm_begin = MPI_Wtime(); 
      hypre_ParVector *r = dmem_all_data->vector_fine.r;
      HYPRE_Real res_norm = sqrt(hypre_ParVectorInnerProd(r, r));
      dmem_all_data->output.residual_norm_wtime += MPI_Wtime() - residual_norm_begin;
      dmem_all_data->iter.cycle += 1;
      if (res_norm/dmem_all_data->output.r0_norm2 < dmem_all_data->input.tol || dmem_all_data->iter.cycle == dmem_all_data->input.num_cycles) break;
   }
   dmem_all_data->output.solve_wtime = MPI_Wtime() - begin;
   hypre_ParCSRMatrixMatvec(1.0,
                            A_array[0],
                            dmem_all_data->vector_fine.x,
                            0.0,
                            dmem_all_data->vector_fine.e);
}

void DMEM_SyncAddCycle(DMEM_AllData *dmem_all_data,
                       HYPRE_Solver solver,
                       HYPRE_Int cycle)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)solver;
   int coarsest_level, coarse_level, fine_level;
   double begin, end;

   hypre_ParVector *x;

   HYPRE_Int num_rows;

   HYPRE_Real *u_local_data;
   HYPRE_Real *v_local_data;
   HYPRE_Real *f_local_data;
   HYPRE_Real *A_diag_data;
   HYPRE_Int *A_diag_i;

   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);

   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   hypre_ParVector *e = dmem_all_data->vector_fine.e;
   HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
   hypre_ParVectorSetConstantValues(e, 0.0);

   x = dmem_all_data->vector_fine.x;
   hypre_ParVectorCopy(dmem_all_data->vector_fine.r,
                       F_array[0]);

   HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(x));

   for (HYPRE_Int level = 0; level < num_levels-1; level++){
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;

      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
                                F_array[fine_grid],
                                0.0,
                                F_array[coarse_grid]);
   }

   for (HYPRE_Int level = 0; level < num_levels; level++){
      if (dmem_all_data->input.solver == SYNC_AFACX && level < num_levels-1){
         coarsest_level = level + 1;
      }
      else {
         coarsest_level = level;
      }

      A_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
      A_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[coarsest_level]));
      num_rows = hypre_ParCSRMatrixNumRows(A_array[coarsest_level]);
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[coarsest_level]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[coarsest_level]));

      if (level == num_levels-1){
         begin = MPI_Wtime();
         hypre_GaussElimSolve(amg_data, coarsest_level, 99);
         dmem_all_data->output.coarsest_solve_wtime += MPI_Wtime() - begin;
      }
      else {
         if (dmem_all_data->input.solver == SYNC_AFACX){
            if (coarsest_level == num_levels-1){
               begin = MPI_Wtime();
               hypre_GaussElimSolve(amg_data, coarsest_level, 99);
               dmem_all_data->output.coarsest_solve_wtime += MPI_Wtime() - begin;
            }
            else {
               hypre_ParVectorCopy(F_array[coarsest_level], Vtemp);
               hypre_ParVectorSetConstantValues(U_array[coarsest_level], 0.0);
               for (int k = 0; k < dmem_all_data->input.num_coarse_smooth_sweeps; k++){
                  for (int i = 0; i < num_rows; i++){
                     u_local_data[i] += dmem_all_data->input.smooth_weight * v_local_data[i] / A_diag_data[A_diag_i[i]];
                  }
                  if (k == dmem_all_data->input.num_coarse_smooth_sweeps-1) break;
                  hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                                     A_array[coarsest_level],
                                                     U_array[coarsest_level],
                                                     1.0,
                                                     F_array[coarsest_level],
                                                     Vtemp);
               }
            }
            fine_level = coarsest_level - 1;
            coarse_level = coarsest_level;
            hypre_ParCSRMatrixMatvec(1.0,
                                     P_array[fine_level],
                                     U_array[coarse_level],
                                     0.0,
                                     U_array[fine_level]);
            hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                               A_array[fine_level],
                                               U_array[fine_level],
                                               1.0,
                                               F_array[fine_level],
                                               e);
            hypre_ParVectorCopy(e, Vtemp);
            u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_level]));
            A_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_level]));
            A_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_level]));
            num_rows = hypre_ParCSRMatrixNumRows(A_array[fine_level]);
            
            hypre_ParVectorSetConstantValues(U_array[fine_level], 0.0);
            for (int k = 0; k < dmem_all_data->input.num_fine_smooth_sweeps; k++){
               for (int i = 0; i < num_rows; i++){
                  u_local_data[i] += dmem_all_data->input.smooth_weight * v_local_data[i] / A_diag_data[A_diag_i[i]];
               }
               if (k == dmem_all_data->input.num_fine_smooth_sweeps-1) break;
               hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                                  A_array[fine_level],
                                                  U_array[fine_level],
                                                  1.0,
                                                  e,
                                                  Vtemp);
            }
         }
         else {
            for (HYPRE_Int i = 0; i < num_rows; i++){
               u_local_data[i] = dmem_all_data->input.smooth_weight * f_local_data[i] / A_diag_data[A_diag_i[i]];
            }

            hypre_ParCSRMatrixMatvec(1.0,
                                     A_array[coarsest_level],
                                     U_array[coarsest_level],
                                     0.0,
                                     Vtemp);

            for (HYPRE_Int i = 0; i < num_rows; i++){
               u_local_data[i] = 2.0 * u_local_data[i] - dmem_all_data->input.smooth_weight * v_local_data[i] / A_diag_data[A_diag_i[i]];
            }
         }
      }

      for (HYPRE_Int interp_level = level; interp_level > 0; interp_level--){
         HYPRE_Int fine_grid = interp_level - 1;
         HYPRE_Int coarse_grid = interp_level;
         hypre_ParCSRMatrixMatvec(1.0,
                                  P_array[fine_grid],
                                  U_array[coarse_grid],
                                  0.0,
                                  U_array[fine_grid]);
      }

      num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
      for (HYPRE_Int i = 0; i < num_rows; i++){
         x_local_data[i] += u_local_data[i];
         //e_local_data[i] += u_local_data[i];
      }
   }
}
