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

   srand(0);
  // srand(omp_get_wtime());

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);

   hypre_ParCSRMatrix *A = A_array[0];
   hypre_ParVector *u = U_array[0];
   hypre_ParVector *f = F_array[0];
   hypre_ParVector *v = hypre_ParAMGDataVtemp(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);

   HYPRE_Real *e = all_data->vector.e[0];
   HYPRE_Real *y = all_data->vector.y[0];

   HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Real *v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   //HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, all_data->input.eig_power_MG_max_iters);
   HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, 1);
   HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, 0);

   for (int i = 0; i < num_rows; i++){ 
      u_local_data[i] = RandDouble(-1.0, 1.0);
      y[i] = f_local_data[i];
   }
   eig_power_iters = 0;
   double start = omp_get_wtime();
   while (1){
      u_norm = sqrt(hypre_ParVectorInnerProd(u, u));
      hypre_ParVectorScale(1.0/u_norm, u);
      //hypre_ParVectorCopy(u, v);
      #pragma omp parallel for
      for (int i = 0; i < num_rows; i++){
         e[i] = u_local_data[i];
      }
      hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, f);
      hypre_ParVectorSetConstantValues(u, 0.0);
      if (all_data->input.solver == MULT){
         HYPRE_BoomerAMGSolve(all_data->hypre.solver, A, f, u);
      }
      else {
         BPXCycle(all_data);
      }

      eig_power_iters++;
      if (eig_power_iters == all_data->input.eig_power_max_iters) break;
   }
   #pragma omp parallel for
   for (int i = 0; i < num_rows; i++){
      v_local_data[i] = e[i];
   }
   eig_max = hypre_ParVectorInnerProd(v, u);

   for (int i = 0; i < num_rows; i++) u_local_data[i] = RandDouble(-1.0, 1.0);
   eig_power_iters = 0;
   while (1){
      u_norm = sqrt(hypre_ParVectorInnerProd(u, u));
      hypre_ParVectorScale(1.0/u_norm, u);
      //hypre_ParVectorCopy(u, v);
      #pragma omp parallel for
      for (int i = 0; i < num_rows; i++){
         e[i] = u_local_data[i];
      }
      hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, f);
      hypre_ParVectorSetConstantValues(u, 0.0);
      if (all_data->input.solver == MULT){
         HYPRE_BoomerAMGSolve(all_data->hypre.solver, A, f, u);
      }
      else {
         BPXCycle(all_data);
      }

      #pragma omp parallel for
      for (int i = 0; i < num_rows; i++){
         v_local_data[i] = e[i];
      }
      eig_power_iters++;
      if (eig_power_iters == all_data->input.eig_power_max_iters) break;

      
      hypre_ParVectorAxpy(-eig_max, v, u);
   }
   eig_min = hypre_ParVectorInnerProd(v, u);
   double eig_time = omp_get_wtime() - start;

   all_data->cheby.beta = eig_max;// - all_data->input.b_eig_shift;
   all_data->cheby.alpha = eig_min;// + all_data->input.a_eig_shift;

   HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, all_data->input.num_cycles);
   HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, all_data->hypre.print_level);
   srand(0);

   if (all_data->input.format_output_flag == 0)
      printf("CHEBY: power eig max %.16e, eig min %.16e, k_2 %.2f, time %e\n",
             all_data->cheby.beta,  all_data->cheby.alpha, eig_max/eig_min, eig_time);

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

   srand(0);
   for (int i = 0; i < num_rows; i++){
      u_local_data[i] = 0.0;
      f_local_data[i] = all_data->vector.y[0][i];
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
