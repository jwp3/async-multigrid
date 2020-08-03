#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Add.hpp"
#include "_hypre_utilities.h"
#include "DMEM_Misc.hpp"
#include "DMEM_Mult.hpp"

int cycle_type;
int precond_zero_init_guess;

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
   precond_zero_init_guess = 1;
   
   double solve_begin = MPI_Wtime();
   while (1){
      DMEM_DelayProc(dmem_all_data);
      DMEM_HypreParVector_Set(dmem_all_data->vector_fine.e, 0.0, num_rows);
      DMEM_MultCycle(dmem_all_data->hypre.solver,
                     dmem_all_data->matrix.A_fine,
                     dmem_all_data->vector_fine.r,
                     dmem_all_data->vector_fine.e);
      DMEM_HypreParVector_Axpy(dmem_all_data->vector_fine.x, 
                               dmem_all_data->vector_fine.e,
                               1.0, num_rows);

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

void DMEM_MultCycle(void *amg_vdata,
                    hypre_ParCSRMatrix *A,
                    hypre_ParVector *f,
                    hypre_ParVector *u)
{
   HYPRE_Int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)amg_vdata;
   double smooth_begin, restrict_begin, prolong_begin, matvec_begin, vecop_begin, level_begin;
   double smooth_end, restrict_end, prolong_end, matvec_end, vecop_end, level_end;

   HYPRE_Real *u_local_data;
   HYPRE_Real *v_local_data;
   HYPRE_Real *f_local_data;
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
   P_array = hypre_ParAMGDataPArray(amg_data);
   R_array = hypre_ParAMGDataRArray(amg_data);

   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

   HYPRE_Int smoother = hypre_ParAMGDataGridRelaxType(amg_data)[1];
   HYPRE_Real *relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
   HYPRE_Int simple = hypre_ParAMGDataSimple(amg_data);

   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   int coarsest_level = num_levels-1;

   if (precond_zero_init_guess == 1){
      DMEM_HypreParVector_Copy(Vtemp, F_array[0], num_rows);      
   }
   else {
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[0],
                                         U_array[0],
                                         1.0,
                                         F_array[0],
                                         Vtemp);
   }

   for (int level = 0; level < coarsest_level; level++){
      level_begin = MPI_Wtime();
      HYPRE_Int fine_grid = level;
      HYPRE_Int coarse_grid = level + 1;

      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A_array[fine_grid]);
      num_rows = hypre_ParCSRMatrixNumRows(A_array[fine_grid]);
      A_data = hypre_CSRMatrixData(A_diag);
      A_i = hypre_CSRMatrixI(A_diag);

      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid]));
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
  
      /* smooth */
      if (smoother == 18){
         DMEM_HypreParVector_Set(U_array[fine_grid], 0.0, num_rows);
         HYPRE_Real *l1_norms = hypre_ParAMGDataL1Norms(amg_data)[level];
         DMEM_HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, l1_norms, num_rows);
      }
      else if (smoother == 0){
        // DMEM_HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, dmem_all_data->matrix.wJacobi_scale_fine[fine_grid], num_rows);
         for (int i = 0; i < num_rows; i++){
            u_local_data[i] += v_local_data[i] * relax_weight[level] / A_data[A_i[i]];
         }
      }
      else {
         hypre_BoomerAMGRelax(A_array[fine_grid],
                              F_array[fine_grid],
                              NULL,
                              3,
                              0,
                              1.0,
                              1.0,
                              NULL,
                              U_array[fine_grid], 
                              Vtemp,
                              Ztemp);
      }

      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[fine_grid],
                                         U_array[fine_grid],
                                         1.0,
                                         F_array[fine_grid],
                                         Vtemp);
      /* restrict */
      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
        			Vtemp,
                                0.0,
                                F_array[coarse_grid]);
      DMEM_HypreParVector_Copy(Vtemp, F_array[coarse_grid], num_rows);
      DMEM_HypreParVector_Set(U_array[coarse_grid], 0.0, num_rows); 
   }

   double coarse_r0_norm2, res_norm;
   
   hypre_GaussElimSolve(amg_data, coarsest_level, 9);
    
   for (int level = coarsest_level; level > 0; level--){
      level_begin = MPI_Wtime();
      HYPRE_Int fine_grid = level - 1;
      HYPRE_Int coarse_grid = level;

      A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
      num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));

      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid]));

      /* prolong and correct */
      hypre_ParCSRMatrixMatvec(1.0,
                               P_array[fine_grid], 
                               U_array[coarse_grid],
                               1.0,
                               U_array[fine_grid]);

      /* smooth */
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[fine_grid],
                                         U_array[fine_grid],
                                         1.0,
                                         F_array[fine_grid],
                                         Vtemp);

      if (smoother == 18){
         HYPRE_Real *l1_norms = hypre_ParAMGDataL1Norms(amg_data)[level];
         DMEM_HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, l1_norms, num_rows);
      }
      else if (smoother == 0){
         for (int i = 0; i < num_rows; i++){
            u_local_data[i] += v_local_data[i] * relax_weight[level] / A_data[A_i[i]];
         }
        // DMEM_HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, dmem_all_data->matrix.wJacobi_scale_fine[fine_grid], num_rows);
      }
      else {
         hypre_BoomerAMGRelax(A_array[fine_grid],
                              F_array[fine_grid],
                              NULL,
                              4,
                              0,
                              1.0,
                              1.0,
                              NULL,
                              U_array[fine_grid],
                              Vtemp,
                              Ztemp);
      }
   }
  // DMEM_HypreParVector_Axpy(u, U_array[0], 1.0, num_rows); 
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
   cycle_type = dmem_all_data->input.solver;

   double begin = MPI_Wtime();
   while(1){
      if (dmem_all_data->input.solver == SYNC_AFACX ||
          dmem_all_data->input.solver == SYNC_AFACJ){
        // DMEM_SyncAFACCycle(dmem_all_data);
         DMEM_SyncAFACCycle(dmem_all_data->hypre.solver,
                            A_array[0],
                            dmem_all_data->vector_fine.r,
                            dmem_all_data->vector_fine.x);
      }
      else {
        // DMEM_SyncAddCycle(dmem_all_data);
         DMEM_SyncAddCycle(dmem_all_data->hypre.solver,
                           A_array[0],
                           dmem_all_data->vector_fine.r,
                           dmem_all_data->vector_fine.x);
      }
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

//void DMEM_SyncAddCycle(DMEM_AllData *dmem_all_data)
void DMEM_SyncAddCycle(void *amg_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector *f,
                       hypre_ParVector *u)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)amg_vdata;
   int coarsest_level, coarse_level, fine_level;

   hypre_ParVector *x;

   HYPRE_Int num_rows;

   HYPRE_Real *u_local_data;
   HYPRE_Real *v_local_data;
   HYPRE_Real *f_local_data;
   HYPRE_Real *A_data;
   HYPRE_Int *A_i;

   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array;
   hypre_ParCSRMatrix **R_array;
   if (cycle_type == SYNC_BPX){
      P_array = amg_data->P_array_afacj;
      R_array = amg_data->P_array_afacj;
   }
   else {
      P_array = hypre_ParAMGDataPArray(amg_data);
      R_array = hypre_ParAMGDataRArray(amg_data);
   }
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);

   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   A_array[0] = A;
   F_array[0] = f;

   num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
   HYPRE_Real *u_accum = (HYPRE_Real *)calloc(num_rows, sizeof(HYPRE_Real));

   HYPRE_Int smoother = hypre_ParAMGDataGridRelaxType(amg_data)[1];
   HYPRE_Real add_rlx_wt = hypre_ParAMGDataAddRelaxWt(amg_data);
   HYPRE_Int simple = hypre_ParAMGDataSimple(amg_data);

  // hypre_ParVector *e = dmem_all_data->vector_fine.e;
  // HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
  // hypre_ParVectorSetConstantValues(e, 0.0);
  // DMEM_HypreParVector_Copy(F_array[0], dmem_all_data->vector_fine.r, num_rows);


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
         A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[level]));
         A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[level]));
         u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));
         f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[level]));
         HYPRE_Real *l1_norms = hypre_ParAMGDataL1Norms(amg_data)[level];
         if (smoother == 18){
            DMEM_HypreParVector_Ivaxpy(U_array[level], F_array[level], l1_norms, num_rows);
         }
         else {
            for (int i = 0; i < num_rows; i++){
               u_local_data[i] = f_local_data[i] * add_rlx_wt / A_data[A_i[i]];
            }
         }

         if (simple == -1){
            hypre_ParCSRMatrixMatvec(1.0,
                                     A_array[coarsest_level],
                                     U_array[coarsest_level],
                                     0.0,
                                     Vtemp);
            DMEM_HypreParVector_Scale(U_array[coarsest_level], 2.0, num_rows);
            if (smoother == 18){
               DMEM_HypreParVector_Ivaxpy(U_array[level], F_array[level], l1_norms, num_rows);
            }
            else {
               for (int i = 0; i < num_rows; i++){
                  u_local_data[i] -= v_local_data[i] * add_rlx_wt / A_data[A_i[i]];
               }
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
     
      num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0])); 
      DMEM_HypreRealArray_Axpy(u_accum, u_local_data, 1.0, num_rows);
     // DMEM_HypreParVector_Axpy(e, U_array[0], 1.0, num_rows);
   }

   num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   DMEM_HypreRealArray_Axpy(u_local_data, u_accum, 1.0, num_rows);

   free(u_accum);
  // DMEM_HypreParVector_Copy(U_array[0], e, num_rows);
}


void DMEM_SyncAFACCycle(void *amg_vdata,
                        hypre_ParCSRMatrix *A,
                        hypre_ParVector *f,
                        hypre_ParVector *u)
{
   HYPRE_Int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)amg_vdata;
   int coarsest_level, coarse_level, fine_level;

   hypre_ParVector *x;

   HYPRE_Int num_rows;

   HYPRE_Real *u_local_data;
   HYPRE_Real *v_local_data;
   HYPRE_Real *f_local_data;
   HYPRE_Real *A_data;
   HYPRE_Int *A_i;

   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_ParCSRMatrix **P_array_afacj = amg_data->P_array_afacj;
   hypre_ParCSRMatrix **R_array_afacj = amg_data->P_array_afacj;
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
   hypre_ParVector *Ztemp = hypre_ParAMGDataZtemp(amg_data);

   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   A_array[0] = A;
   F_array[0] = f;

   num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
   HYPRE_Real *u_accum = (HYPRE_Real *)calloc(num_rows, sizeof(HYPRE_Real));

   HYPRE_Int smoother = hypre_ParAMGDataGridRelaxType(amg_data)[1];
   HYPRE_Real add_rlx_wt = hypre_ParAMGDataAddRelaxWt(amg_data); 
   HYPRE_Int simple = hypre_ParAMGDataSimple(amg_data);

  // hypre_ParVector *e = dmem_all_data->vector_fine.e;
  // HYPRE_Real *e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(e));
  // hypre_ParVectorSetConstantValues(e, 0.0);

   for (int level = 0; level < num_levels; level++){
      for (int interp_level = 0; interp_level < level; interp_level++){
         HYPRE_Int fine_grid = interp_level;
         HYPRE_Int coarse_grid = interp_level + 1;

         hypre_ParCSRMatrix *R;
         if (fine_grid == level-1 &&
             cycle_type == SYNC_AFACJ){
            R = R_array[fine_grid];
         }
         else {
            R = R_array_afacj[fine_grid];
         }

         hypre_ParCSRMatrixMatvecT(1.0,
                                   R,
                                   F_array[fine_grid],
                                   0.0,
                                   F_array[coarse_grid]);
      }

      if (level == num_levels-1){
         hypre_GaussElimSolve(amg_data, level, 99);
      }
      else {
         hypre_ParVectorSetConstantValues(U_array[level], 0.0);
         int num_smooth_sweeps = 2;
         for (int i = 0; i < num_smooth_sweeps; i++){
            hypre_BoomerAMGRelax(A_array[level],
                                 F_array[level],
                                 NULL,
                                 0,
                                 0,
                                 add_rlx_wt,
                                 1.0,
                                 NULL,
                                 U_array[level],
                                 Vtemp,
                                 Ztemp);
         }
//         A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[level]));
//         A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[level]));
//         num_rows = hypre_ParCSRMatrixNumRows(A_array[level]);
//         u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));
//         f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[level]));
//         HYPRE_Real *l1_norms = hypre_ParAMGDataL1Norms(amg_data)[level];
//         if (smoother == 18){
//            DMEM_HypreParVector_Ivaxpy(U_array[level], F_array[level], l1_norms, num_rows);
//         }
//         else {
//            for (int i = 0; i < num_rows; i++){
//               u_local_data[i] = f_local_data[i] * add_rlx_wt / A_data[A_i[i]];
//            }
//         }
//        // if (smoother == 18){
//        //    DMEM_HypreParVector_Ivaxpy(U_array[level], F_array[level], dmem_all_data->matrix.L1_row_norm_fine[level], num_rows);
//        // }
//        // else {
//        //    DMEM_HypreParVector_Ivaxpy(U_array[level], F_array[level], dmem_all_data->matrix.wJacobi_scale_fine[level], num_rows);
//        // }
//
//         
//         if (simple == -1){
//            hypre_ParCSRMatrixMatvec(1.0,
//                                     A_array[level],
//                                     U_array[level],
//                                     0.0,
//                                     Vtemp);
//            DMEM_HypreParVector_Scale(U_array[level], 2.0, num_rows);
//            if (smoother == 18){
//               DMEM_HypreParVector_Ivaxpy(U_array[level], Vtemp, l1_norms, num_rows);
//            }
//            else {
//               for (int i = 0; i < num_rows; i++){
//                  u_local_data[i] -= v_local_data[i] * add_rlx_wt / A_data[A_i[i]];
//               }
//            }
//           // if (smoother == L1_JACOBI){
//           //    DMEM_HypreParVector_Ivaxpy(U_array[level], Vtemp, dmem_all_data->matrix.symmL1_row_norm_fine[level], num_rows);
//           // }
//           // else {
//           //    DMEM_HypreParVector_Ivaxpy(U_array[level], Vtemp, dmem_all_data->matrix.symmwJacobi_scale_fine[level], num_rows);
//           // }
//         }
      }

      for (int interp_level = level; interp_level > 0; interp_level--){
         HYPRE_Int fine_grid = interp_level - 1;
         HYPRE_Int coarse_grid = interp_level;
         
         hypre_ParCSRMatrix *P;
         if (fine_grid == level-1){
            P = P_array[fine_grid];
         }
         else {
            P = P_array_afacj[fine_grid];
         }
 
         hypre_ParCSRMatrixMatvec(1.0,
                                  P,
                                  U_array[coarse_grid],
                                  0.0,
                                  U_array[fine_grid]);
      }

      num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
      DMEM_HypreRealArray_Axpy(u_accum, u_local_data, 1.0, num_rows);
     // DMEM_HypreParVector_Axpy(e, U_array[0], 1.0, num_rows);
   }

   num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);
   u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   DMEM_HypreRealArray_Axpy(u_local_data, u_accum, 1.0, num_rows);

   free(u_accum);
}
