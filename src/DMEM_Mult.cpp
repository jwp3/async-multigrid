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

   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   double vecop_begin, matvec_begin, residual_norm_begin;
   double vecop_end, matvec_end, residual_norm_end;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[0]); 

   
   DMEM_HypreParVector_Copy(U_array[0], dmem_all_data->vector_fine.x, num_rows);
   DMEM_HypreParVector_Copy(F_array[0], dmem_all_data->vector_fine.b, num_rows);
  
   dmem_all_data->iter.cycle = 0;
   
   double solve_begin = MPI_Wtime();
   while (1){
      if (dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL){
         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Set(U_array[0], 0.0, num_rows);
         DMEM_HypreParVector_Copy(F_array[0], dmem_all_data->vector_fine.r, num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
      }

      DMEM_DelayProc(dmem_all_data);
      DMEM_MultCycle(dmem_all_data);

      if (dmem_all_data->input.accel_type == CHEBY_ACCEL || dmem_all_data->input.accel_type == RICHARD_ACCEL){
         DMEM_ChebyUpdate(dmem_all_data, dmem_all_data->vector_fine.d, U_array[0], num_rows);         

         vecop_begin = MPI_Wtime();
         DMEM_HypreParVector_Axpy(dmem_all_data->vector_fine.x, dmem_all_data->vector_fine.d, 1.0, num_rows);
         dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

        // vecop_begin = MPI_Wtime();
        // DMEM_HypreParVector_Axpy(dmem_all_data->vector_fine.x, U_array[0], 1.0, num_rows);
        // dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;

        // hypre_ParVector *d = dmem_all_data->vector_fine.d;
        // HYPRE_Real d_norm = sqrt(hypre_ParVectorInnerProd(d, d));
        // HYPRE_Real u_norm = sqrt(hypre_ParVectorInnerProd(U_array[0], U_array[0]));
        // if (my_id == 0) printf("%e %e\n", d_norm, u_norm);
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
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
   hypre_ParVector *Ztemp = hypre_ParAMGDataZtemp(amg_data);

   hypre_ParCSRMatrix **P_array;
   hypre_ParCSRMatrix **R_array;
   if (dmem_all_data->input.solver == MULT){
      P_array = hypre_ParAMGDataPArray(amg_data);
      R_array = hypre_ParAMGDataRArray(amg_data);
   }
   else {
      P_array = amg_data->P_array_afacj;
      R_array = amg_data->P_array_afacj;
   }

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
   double vecop_begin, vecop_end;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);

   dmem_all_data->iter.cycle = 0;

   double begin = MPI_Wtime();
   while(1){
      DMEM_SyncAddCycle(dmem_all_data);
      vecop_begin = MPI_Wtime();
      DMEM_HypreParVector_Axpy(dmem_all_data->vector_fine.x, U_array[0], 1.0, num_rows);
      dmem_all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
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

void DMEM_SyncAddCycle(DMEM_AllData *dmem_all_data)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   int coarsest_level, coarse_level, fine_level;

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


   num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
   DMEM_HypreParVector_Copy(F_array[0], dmem_all_data->vector_fine.r, num_rows);

   for (int level = 0; level < num_levels-1; level++){
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;

      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
                                F_array[fine_grid],
                                0.0,
                                F_array[coarse_grid]);
   }

   for (int level = 0; level < num_levels; level++){
      coarsest_level = level;
      num_rows = hypre_ParCSRMatrixNumRows(A_array[coarsest_level]);

      if (level == num_levels-1){
         hypre_GaussElimSolve(amg_data, coarsest_level, 99);
      }
      else {
         DMEM_HypreParVector_Set(U_array[coarsest_level], 0.0, num_rows);
         if (dmem_all_data->input.smoother == L1_JACOBI){
            DMEM_HypreParVector_Ivaxpy(U_array[coarsest_level], F_array[coarsest_level], dmem_all_data->matrix.L1_row_norm_fine[coarsest_level], num_rows);
         }
         else {
            DMEM_HypreParVector_Ivaxpy(U_array[coarsest_level], F_array[coarsest_level], dmem_all_data->matrix.wJacobi_scale_fine[coarsest_level], num_rows);
         }

         if (dmem_all_data->input.simple_jacobi_flag == 0){
            hypre_ParCSRMatrixMatvec(1.0,
                                     A_array[coarsest_level],
                                     U_array[coarsest_level],
                                     0.0,
                                     Vtemp);
            DMEM_HypreParVector_Scale(U_array[coarsest_level], 2.0, num_rows);
            if (dmem_all_data->input.smoother == L1_JACOBI){
               DMEM_HypreParVector_Ivaxpy(U_array[coarsest_level], Vtemp, dmem_all_data->matrix.symmL1_row_norm_fine[coarsest_level], num_rows);
            }
            else {
               DMEM_HypreParVector_Ivaxpy(U_array[coarsest_level], Vtemp, dmem_all_data->matrix.symmwJacobi_scale_fine[coarsest_level], num_rows);
            }
         }
      }

      for (int interp_level = level; interp_level > 0; interp_level--){
         HYPRE_Int fine_grid = interp_level - 1;
         HYPRE_Int coarse_grid = interp_level;
         hypre_ParCSRMatrixMatvec(1.0,
                                  P_array[fine_grid],
                                  U_array[coarse_grid],
                                  0.0,
                                  U_array[fine_grid]);
      }

      DMEM_HypreParVector_Axpy(e, U_array[0], 1.0, num_rows);
   }

   DMEM_HypreParVector_Copy(U_array[0], e, num_rows);
}
