#include "Main.hpp"
#include "Misc.hpp"
#include "SMEM_MatVec.hpp"

using namespace std;

void BPXCycle(AllData *all_data);

void ChebySetup(AllData *all_data)
{
   if (all_data->input.eig_power_max_iters <= 0) return;
   int eig_power_iters;
   double eig_max, eig_min;
   HYPRE_Real u_norm, y_norm;
   int num_threads = all_data->input.num_threads;

  // srand((double)my_id * MPI_Wtime());
   srand(omp_get_wtime());

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);

   hypre_ParCSRMatrix *A = A_array[0];
   hypre_ParVector *u = U_array[0];
   hypre_ParVector *f = F_array[0];
   hypre_ParVector *v = hypre_ParAMGDataVtemp(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);

   HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(f));

   //HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, all_data->input.eig_power_MG_max_iters);
   HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, 1);
   HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, 0);

   for (int i = 0; i < num_rows; i++) u_local_data[i] = RandDouble(0.0, 1.0)-.5;
   eig_power_iters = 0;
   while (1){
      u_norm = sqrt(hypre_ParVectorInnerProd(u, u));
      hypre_ParVectorScale(1.0/u_norm, u);
      hypre_ParVectorCopy(u, v);
      hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, f);
      hypre_ParVectorSetConstantValues(u, 0.0);
      BPXCycle(all_data);
     // HYPRE_BoomerAMGSolve(all_data->hypre.solver, A, f, u);

      eig_power_iters++;
      if (eig_power_iters == all_data->input.eig_power_max_iters) break;
   }
   eig_max = hypre_ParVectorInnerProd(v, u);

   for (int i = 0; i < num_rows; i++) u_local_data[i] = RandDouble(0.0, 1.0)-.5;
   eig_power_iters = 0;
   while (1){
      u_norm = sqrt(hypre_ParVectorInnerProd(u, u));
      hypre_ParVectorScale(1.0/u_norm, u);
      hypre_ParVectorCopy(u, v);
      hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, f);
      hypre_ParVectorSetConstantValues(u, 0.0);
      BPXCycle(all_data);
     // HYPRE_BoomerAMGSolve(all_data->hypre.solver, A, f, u);

      eig_power_iters++;
      if (eig_power_iters == all_data->input.eig_power_max_iters) break;

      hypre_ParVectorAxpy(-eig_max, v, u);
   }
   eig_min = hypre_ParVectorInnerProd(v, u);

   all_data->cheby.beta = eig_max;// - all_data->input.b_eig_shift;
   all_data->cheby.alpha = eig_min;// + all_data->input.a_eig_shift;

   HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, all_data->input.num_cycles);
   HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, all_data->hypre.print_level);
   srand(0);
  // if (all_data->input.oneline_output_flag == 0)
      printf("CHEBY: power eig max %.16e, eig min %.16e\n",
             all_data->cheby.beta,  all_data->cheby.alpha);

   double alpha = all_data->cheby.alpha;
   double beta = all_data->cheby.beta;
   all_data->cheby.mu = (beta + alpha) / (beta - alpha);
   all_data->cheby.delta = 2.0 / (beta + alpha);

   double omega = 2.0 / (1.0 + sqrt(1.0 - pow(all_data->cheby.mu, -2.0)));
   int N = hypre_CSRMatrixNumRows(all_data->matrix.AA);
   all_data->cheby.c_prev = (double *)malloc(num_threads * sizeof(double));
   all_data->cheby.c = (double *)malloc(num_threads * sizeof(double));
   all_data->cheby.omega = (double *)malloc(num_threads * sizeof(double));
   for (int t = 0; t < num_threads; t++){
      all_data->cheby.c_prev[t] = 1.0;
      all_data->cheby.c[t] = all_data->cheby.mu;
      all_data->cheby.omega[t] = omega;
   }

   for (int i = 0; i < num_rows; i++){
      u_local_data[i] = 0.0;
      f_local_data[i] = 1.0;
   }  
}

void BPXCycle(AllData *all_data)
{
   int num_rows;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector *v = hypre_ParAMGDataVtemp(amg_data);


   HYPRE_Real *f_local_data, *u_local_data, *v_local_data, *A_data;
   HYPRE_Int *A_i;

  // num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
  // v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(v));
  // #pragma omp parallel for
  // for (int i = 0; i < num_rows; i++){
  //    v_local_data[i] = 0.0;
  // }

   for (int level = 0; level < all_data->grid.num_levels; level++){   
      for (int inner_level = 0; inner_level < level; inner_level++){
         int fine_grid = inner_level;
         int coarse_grid = inner_level + 1;

         hypre_ParCSRMatrixMatvecT(1.0,
                                   R_array[fine_grid],
                                   F_array[fine_grid],
                                   0.0,
                                   F_array[coarse_grid]);
        // #pragma omp parallel
        // SMEM_Sync_Parfor_MatVec(all_data,
        //                         hypre_ParCSRMatrixDiag(R_array[fine_grid]),
        //                         hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid])),
        //                         hypre_VectorData(hypre_ParVectorLocalVector(F_array[coarse_grid])));
      }

      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A_array[level]);
      A_data = hypre_CSRMatrixData(A_diag);
      A_i = hypre_CSRMatrixI(A_diag);
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[level]));
      num_rows = hypre_CSRMatrixNumRows(A_diag);
      #pragma omp parallel for
      for (int i = 0; i < num_rows; i++){
         u_local_data[i] = all_data->input.smooth_weight * f_local_data[i] / A_data[A_i[i]];
      }

      for (int inner_level = level-1; inner_level >= 0; inner_level--){
         int fine_grid = inner_level;
         int coarse_grid = inner_level + 1;
         
        // #pragma omp parallel
        // SMEM_Sync_Parfor_MatVec(all_data, 
        //                         hypre_ParCSRMatrixDiag(P_array[fine_grid]),
        //                         hypre_VectorData(hypre_ParVectorLocalVector(U_array[coarse_grid])),
        //                         hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid])));

         double alpha = 0.0;
         if (fine_grid == 0){
            alpha = 1.0;
         }
         hypre_ParCSRMatrixMatvec(1.0,
                                  P_array[fine_grid],
                                  U_array[coarse_grid],
                                  alpha,
                                  U_array[fine_grid]);
      }

     // num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));
     // u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
     // #pragma omp parallel for
     // for (int i = 0; i < num_rows; i++){
     //    v_local_data[i] += u_local_data[i];
     // }
   }
   
  // #pragma omp parallel for
  // for (int i = 0; i < num_rows; i++){
  //    u_local_data[i] = v_local_data[i];
  // }
}

//void MultCycle(AllData *all_data)
//{
//   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
//   double smooth_begin, restrict_begin, prolong_begin, matvec_begin, vecop_begin, level_begin;
//   double smooth_end, restrict_end, prolong_end, matvec_end, vecop_end, level_end;
//
//   HYPRE_Real *u_local_data;
//   HYPRE_Real *v_local_data;
//   HYPRE_Real *f_local_data;
//   HYPRE_Real *r_local_data;
//   HYPRE_Real *x_local_data;
//   HYPRE_Real *A_data;
//   HYPRE_Int *A_i;
//   HYPRE_Int num_rows;
//
//   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
//   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
//   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
//   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
//   hypre_ParVector *Ztemp = hypre_ParAMGDataZtemp(amg_data);
//
//   hypre_ParCSRMatrix **P_array;
//   hypre_ParCSRMatrix **R_array;
//
//   if (all_data->input.solver == MULT){
//      P_array = hypre_ParAMGDataPArray(amg_data);
//      R_array = hypre_ParAMGDataRArray(amg_data);
//   }
//   else {
//      P_array = amg_data->P_array_afacj;
//      R_array = amg_data->P_array_afacj;
//   }
//
//   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
//   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
//
//   int coarsest_level = all_data->grid.num_levels-1;
//   if (all_data->input.solver == MULT_MULTADD){
//      coarsest_level = all_data->input.coarsest_mult_level;
//   }
//
//   for (HYPRE_Int level = 0; level < coarsest_level; level++){
//      level_begin = MPI_Wtime();
//      HYPRE_Int fine_grid = level;
//      HYPRE_Int coarse_grid = level + 1;
//
//      hypre_ParCSRMatrix *A = A_array[fine_grid];
//
//      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
//      num_rows = hypre_ParCSRMatrixNumRows(A);
//      A_data = hypre_CSRMatrixData(A_diag);
//      A_i = hypre_CSRMatrixI(A_diag);
//
//      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid]));
//      r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(all_data->vector_fine.r));
//      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
//
//      smooth_begin = vecop_begin = MPI_Wtime();
//      if (level == 0){
//         HypreParVector_Copy(Vtemp, all_data->vector_fine.r, num_rows);
//      }
//      else {
//         HypreParVector_Copy(Vtemp, F_array[fine_grid], num_rows);
//         HypreParVector_Set(U_array[fine_grid], 0.0, num_rows);
//      }
//      all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
//
//      if (all_data->input.smoother == L1_JACOBI){
//         vecop_begin = MPI_Wtime();
//         HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, all_data->matrix.L1_row_norm_fine[fine_grid], num_rows);
//         all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
//      }
//      else if (all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL ||
//               all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL){
//         hypre_BoomerAMGRelax(A,
//                              F_array[fine_grid],
//                              NULL,
//                              3,
//                              0,
//                              1.0,
//                              1.0,
//                              NULL,
//                              U_array[fine_grid],
//                              all_data->vector_fine.e,
//                              Ztemp);
//      }
//      else {
//         vecop_begin = MPI_Wtime();
//         HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, all_data->matrix.wJacobi_scale_fine[fine_grid], num_rows);
//         all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
//      }
//
//      vecop_begin = MPI_Wtime();
//      HypreParVector_Copy(all_data->vector_fine.e, F_array[fine_grid], num_rows);
//      all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
//
//      matvec_begin = MPI_Wtime();
//      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
//                                         A_array[fine_grid],
//                                         U_array[fine_grid],
//                                         1.0,
//                                         all_data->vector_fine.e,
//                                         Vtemp);
//      all_data->output.smooth_wtime += MPI_Wtime() - smooth_begin;
//      /* restrict */
//      restrict_begin = MPI_Wtime();
//      hypre_ParCSRMatrixMatvecT(1.0,
//                                R_array[fine_grid],
//                                Vtemp,
//                                0.0,
//                                F_array[coarse_grid]);
//      level_end = MPI_Wtime();
//      all_data->output.matvec_wtime += level_end - matvec_begin;
//      all_data->output.restrict_wtime += level_end - restrict_begin;
//      all_data->output.level_wtime[level] += level_end - level_begin;
//   }
//
//   double coarse_r0_norm2, res_norm;
//
//   smooth_begin = MPI_Wtime();
//   hypre_GaussElimSolve(amg_data, coarsest_level, 9);
//   all_data->output.smooth_wtime += MPI_Wtime() - smooth_begin;
//
//   for (HYPRE_Int level = coarsest_level; level > 0; level--){
//      level_begin = MPI_Wtime();
//      HYPRE_Int fine_grid = level - 1;
//      HYPRE_Int coarse_grid = level;
//
//      vecop_begin = MPI_Wtime();
//      HypreParVector_Copy(all_data->vector_fine.e, U_array[fine_grid], num_rows);
//      all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
//
//      /* prolong and correct */
//      prolong_begin = matvec_begin = MPI_Wtime();
//      hypre_ParCSRMatrixMatvecOutOfPlace(1.0,
//                                         P_array[fine_grid],
//                                         U_array[coarse_grid],
//                                         1.0,
//                                         all_data->vector_fine.e,
//                                         U_array[fine_grid]);
//      prolong_end = matvec_end = MPI_Wtime();
//      all_data->output.matvec_wtime += matvec_end - matvec_begin;
//      all_data->output.prolong_wtime += prolong_end - prolong_begin;
//
//      /* smooth */
//      hypre_ParCSRMatrix *A = A_array[fine_grid];
//      A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
//      A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
//      num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[fine_grid]));
//
//      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[fine_grid]));
//      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[fine_grid]));
//
//      vecop_begin = MPI_Wtime();
//      HypreParVector_Copy(all_data->vector_fine.e, F_array[fine_grid], num_rows);
//      all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
//
//      /* smooth */
//      matvec_begin = smooth_begin = MPI_Wtime();
//      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
//                                         A_array[fine_grid],
//                                         U_array[fine_grid],
//                                         1.0,
//                                         all_data->vector_fine.e,
//                                         Vtemp);
//      all_data->output.matvec_wtime += MPI_Wtime() - matvec_begin;
//
//      if (all_data->input.smoother == L1_JACOBI){
//         vecop_begin = MPI_Wtime();
//         HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, all_data->matrix.L1_row_norm_fine[fine_grid], num_rows);
//         all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
//      }
//      else {
//         vecop_begin = MPI_Wtime();
//         HypreParVector_Ivaxpy(U_array[fine_grid], Vtemp, all_data->matrix.wJacobi_scale_fine[fine_grid], num_rows);
//         all_data->output.vecop_wtime += MPI_Wtime() - vecop_begin;
//      }
//      all_data->output.smooth_wtime += MPI_Wtime() - smooth_begin;
//      all_data->output.level_wtime[level] += MPI_Wtime() - level_begin;
//   }
//}
//}
