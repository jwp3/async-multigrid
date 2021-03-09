#include "Main.hpp"
#include "Misc.hpp"
#include "SEQ_MatVec.hpp"
#include "SEQ_AMG.hpp"
#include "SMEM_MatVec.hpp"
#include "SMEM_Sync_AMG.hpp"
#include "SMEM_Async_AMG.hpp"
#include "SMEM_Smooth.hpp"
#include "SEQ_Smooth.hpp"

void SMEM_Solve(AllData *all_data)
{
   double delta = all_data->cheby.delta;
   double mu = all_data->cheby.mu;
   double mu22 = pow(2.0 * mu, 2.0);
   double sum, r_inner_prod;
   HYPRE_Real *u_outer, *y_outer;
   HYPRE_Int *A_i = hypre_CSRMatrixI(all_data->matrix.A[0]);
   HYPRE_Real *A_data = hypre_CSRMatrixData(all_data->matrix.A[0]);
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector *v = hypre_ParAMGDataVtemp(amg_data);
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));
   HYPRE_Real *v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(v));
   HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, 0);
   HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, 1);

   int num_threads = all_data->input.num_threads;
   unsigned int *usec_vec = (unsigned int *)calloc(num_threads, sizeof(unsigned int));
   srand(0);
   int num_delayed = (int)ceil((double)num_threads * all_data->input.delay_frac);
   int count_delayed = 0;
   for (int t = num_threads-1; t >= 0; t--){
      if (count_delayed == num_delayed) break;
      count_delayed++;
      usec_vec[t] = all_data->input.delay_usec;
   }

   HYPRE_Real *r;
   if (all_data->input.solver == PAR_BPX){
      r = &(all_data->vector.rr[0]);
   }
   else {
      r = all_data->vector.r[0];
   }

   if (all_data->input.precond_flag == 1){
      u_outer = (HYPRE_Real *)calloc(all_data->grid.n[0], sizeof(HYPRE_Real));      
      y_outer = (HYPRE_Real *)calloc(all_data->grid.n[0], sizeof(HYPRE_Real));
   }

   int k_start = 0;

   #pragma omp parallel
   {
      SMEM_Sync_Residual(all_data,
                                all_data->matrix.A[0],
                                all_data->vector.f[0],
                                all_data->vector.u[0],
                                all_data->vector.y[0],
                                r);
   }
   all_data->output.r0_norm2 = Parfor_Norm2(r, all_data->grid.n[0]);

   double start;
   if (all_data->input.async_flag == 1){
      start = omp_get_wtime();
      if (all_data->input.num_threads == 1){
         SEQ_Add_Vcycle_SimRand(all_data);
	 all_data->output.solve_wtime = omp_get_wtime() - start;
      }
      else {
         SMEM_Async_Add_AMG(all_data);
      }
      #pragma omp parallel
      {
         SMEM_Sync_Residual(all_data,
                                   all_data->matrix.A[0],
                                   all_data->vector.f[0],
                                   all_data->vector.u[0],
                                   all_data->vector.y[0],
                                   r);
      }
      all_data->output.r_norm2 = Parfor_Norm2(r, all_data->grid.n[0]);
   }
   else{
      double *r_norm_loc = (double *)calloc(all_data->input.num_threads, sizeof(double));
      if (all_data->input.print_reshist_flag == 1 &&
          all_data->input.format_output_flag == 0){
         printf("\nIters\tRel. Res. 2-norm\n"
                "-----\t----------------\n");
      }
      if (all_data->input.print_reshist_flag == 1 &&
          all_data->input.format_output_flag == 0){
         printf("%d\t%e\n", 0, 1.);
      }
      if (all_data->input.thread_part_type == ALL_LEVELS){ 
         omp_init_lock(&(all_data->thread.lock));
      }
      start = omp_get_wtime();
      #pragma omp parallel
      {
         int tid = omp_get_thread_num();
         double omega = 2.0;
         srand(tid);
         unsigned int usec = 0;
         int delay_flag = 0;
         int delay_thread = num_threads-1;
         if (all_data->input.delay_flag == DELAY_SOME || all_data->input.delay_flag == DELAY_ALL){
            delay_flag = 1;
            usec = usec_vec[tid];
         }
         else if (tid == delay_thread && all_data->input.delay_flag == DELAY_ONE){
            delay_flag = 1;
            usec = all_data->input.delay_usec;
         }
         else if (tid == delay_thread && all_data->input.delay_flag == FAIL_ONE){
            usec = all_data->input.delay_usec;
         }

         for (int k = k_start; k <= all_data->input.num_cycles; k++){
            if (tid == delay_thread && all_data->input.delay_flag == FAIL_ONE){
               if (k == all_data->input.fail_iter){
                  delay_flag = 1;
               }
               else {
                  delay_flag = 0;
               }
            }
            
            if (delay_flag == 1){
#ifdef WINDOWS
               this_thread::sleep_for(chrono::microseconds(usec));
#else
               usleep(usec);
#endif
            }

            if (all_data->input.solver == MULTADD){
               if (all_data->input.thread_part_type == ALL_LEVELS){
                  SMEM_Sync_Add_Vcycle(all_data);
               }
               else{
               }
            }
            else if (all_data->input.solver == AFACX){
               if (all_data->input.thread_part_type == ALL_LEVELS){
                  SMEM_Sync_Add_Vcycle(all_data);
               }
               else{
                  SMEM_Sync_Parfor_AFACx_Vcycle(all_data);
               }
            }
            else if (all_data->input.solver == BPX || all_data->input.solver == PAR_BPX){
               SMEM_Sync_Parfor_BPXcycle(all_data);
            }
            else{
               SMEM_Sync_Parfor_Vcycle(all_data);
            }

            if (all_data->input.cheby_flag == 1){
               if (k == 0){
                  #pragma omp for
                  for (int i = 0; i < all_data->grid.n[0]; i++){
                     u_outer[i] = delta * all_data->vector.u[0][i];
                     y_outer[i] = 0;
                     all_data->vector.u[0][i] = u_outer[i];
                  }
               }
               else {
                  #pragma omp for
                  for (int i = 0; i < all_data->grid.n[0]; i++){
                     double u_outer_prev = u_outer[i];
                     u_outer[i] = y_outer[i] + omega * (delta * all_data->vector.u[0][i] + u_outer[i] - y_outer[i]);
                     y_outer[i] = u_outer_prev;
                     all_data->vector.u[0][i] = u_outer[i];
                  }
               }
               omega = 1.0 / (1.0 - omega / mu22);
            }
 
            double residual_start = omp_get_wtime();
            if (tid == 0) r_inner_prod = 0;
            SMEM_Sync_Residual(all_data,
                                      all_data->matrix.A[0],
                                      all_data->vector.f[0],
                                      all_data->vector.u[0],
                                      all_data->vector.y[0],
                                      r);
            if (all_data->input.check_resnorm_flag == 1){
               #pragma omp for reduction(+:r_inner_prod)
               for (int i = 0; i < all_data->grid.n[0]; i++){
                   r_inner_prod += r[i] * r[i];
               }
               double r_norm2 = sqrt(r_inner_prod);
              // r_norm_loc[tid] = 0;
              // #pragma omp for
              // for (int i = 0; i < all_data->grid.n[0]; i++){
              //    r_norm_loc[tid] += pow(r[i], 2.0);
              // }
              // if (tid == 0){
              //    sum = 0;
              //    for (int t = 0; t < all_data->input.num_threads; t++){
              //       sum += r_norm_loc[t];
              //    }
              //    all_data->output.r_norm2 = sqrt(sum);
              // }
              // #pragma omp barrier
               all_data->output.residual_wtime[tid] += omp_get_wtime() - residual_start;
               if (tid == 0){
                  all_data->output.num_cycles = k;
                  all_data->output.r_norm2 = r_norm2;
               }
               if (r_norm2/all_data->output.r0_norm2 < all_data->input.tol){
                  break;
               }
            }
            else {
               all_data->output.residual_wtime[tid] += omp_get_wtime() - residual_start;
               if (tid == 0){
                  all_data->output.num_cycles = k;
               }
            }
            if (all_data->input.print_reshist_flag && tid == 0){
               printf("%d\t%e\n", k+1, all_data->output.r_norm2/all_data->output.r0_norm2);
            }
         }
      }
      if (all_data->input.thread_part_type == ALL_LEVELS){
         omp_destroy_lock(&(all_data->thread.lock));
      }
      all_data->output.solve_wtime = omp_get_wtime() - start;
      for (int level = 0; level < all_data->grid.num_levels; level++){
         all_data->grid.local_num_correct[level] = all_data->output.num_cycles;
      }
      if (all_data->input.check_resnorm_flag == 0){
          all_data->output.r_norm2 = Parfor_Norm2(r, all_data->grid.n[0]); 
      }
   }

   if (all_data->input.precond_flag == 1){
      free(u_outer);
      free(y_outer);
   }
}

void SMEM_Smooth(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *f,
                 HYPRE_Real *u,
                 HYPRE_Real *y,
                 HYPRE_Real *r,
                 int num_sweeps,
                 int level,
                 int ns, int ne)
{
   int tid = omp_get_thread_num();
   //if (all_data->input.num_threads > 1){
      if (all_data->input.thread_part_type == ALL_LEVELS){
         if (all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL){
            SMEM_Sync_HybridJacobiGaussSeidel(all_data, A, f, u, y, num_sweeps, level, ns, ne);
         }
         else if (all_data->input.smoother == ASYNC_GAUSS_SEIDEL){
            SMEM_Async_GaussSeidel(all_data, A, f, u, num_sweeps, level, ns, ne);
         }
         else if (all_data->input.smoother == SEMI_ASYNC_GAUSS_SEIDEL){
            SMEM_SemiAsync_GaussSeidel(all_data, A, f, u, num_sweeps, level, ns, ne);
         }
         else if (all_data->input.smoother == L1_JACOBI){
	    if (all_data->input.solver == MULTADD ||
                all_data->input.solver == ASYNC_MULTADD){
               if (all_data->input.num_post_smooth_sweeps > 0 &&
                   all_data->input.num_pre_smooth_sweeps > 0){
                  SMEM_Sync_SymmetricL1Jacobi(all_data, A, f, u, y, r, num_sweeps, level, ns, ne);
               }
               else{
		 // SMEM_Async_GaussSeidel(all_data, A, f, u, num_sweeps, level, ns, ne);
	          SMEM_Sync_L1Jacobi(all_data, A, f, u, y, num_sweeps, level, ns, ne);
	         // SMEM_Sync_SymmetricL1Jacobi(all_data, A, f, u, y, r, num_sweeps, level, ns, ne);
               }
            }
            else{
               SMEM_Sync_L1Jacobi(all_data, A, f, u, y, num_sweeps, level, ns, ne);
            }
         }
         else {
	    if (all_data->input.solver == MULTADD ||
                all_data->input.solver == ASYNC_MULTADD){
	       if (all_data->input.num_post_smooth_sweeps > 0 &&
                   all_data->input.num_pre_smooth_sweeps > 0){
                  SMEM_Sync_SymmetricJacobi(all_data, A, f, u, y, r, num_sweeps, level, ns, ne);
               }
               else{
	         // SMEM_Async_GaussSeidel(all_data, A, f, u, num_sweeps, level, ns, ne);
	          SMEM_Sync_Jacobi(all_data, A, f, u, y, num_sweeps, level, ns, ne);
	         // SMEM_Async_Parfor_Jacobi(all_data, A, f, u, y, num_sweeps, level);
	         // SMEM_Sync_SymmetricJacobi(all_data, A, f, u, y, r, num_sweeps, level, ns, ne);
               }
	    }
	    else{
	       SMEM_Sync_Jacobi(all_data, A, f, u, y, num_sweeps, level, ns, ne);
	    }
         }
      }
      else{
         if (all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL){
            SMEM_Sync_Parfor_HybridJacobiGaussSeidel(all_data, A, f, u, y, num_sweeps, level);
         }
         else if (all_data->input.smoother == ASYNC_GAUSS_SEIDEL){
            SMEM_Async_Parfor_GaussSeidel(all_data, A, f, u, num_sweeps, level);
         }
         else if (all_data->input.smoother == SEMI_ASYNC_GAUSS_SEIDEL){
            SMEM_SemiAsync_Parfor_GaussSeidel(all_data, A, f, u, num_sweeps, level);
         }
         else if (all_data->input.smoother == L1_JACOBI){
            SMEM_Sync_Parfor_L1Jacobi(all_data, A, f, u, y, num_sweeps, level);
         }
         else {
            SMEM_Sync_Parfor_Jacobi(all_data, A, f, u, y, num_sweeps, level);
         }
      }
  // }
  // else{
  //    if (all_data->input.smoother == GAUSS_SEIDEL ||
  //        all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL){
  //       SEQ_GaussSeidel(all_data, A, f, u, num_sweeps);
  //    }
  //    else if (all_data->input.smoother == L1_JACOBI){
  //       if ((all_data->input.solver == MULTADD ||
  //            all_data->input.solver == ASYNC_MULTADD)
  //            &&
  //           (all_data->input.num_post_smooth_sweeps > 0 &&
  //            all_data->input.num_pre_smooth_sweeps > 0)){
  //          SEQ_SymmetricL1Jacobi(all_data, A, f, u, y, r, num_sweeps, level);
  //       }
  //       else{
  //          SEQ_L1Jacobi(all_data, A, f, u, y, all_data->matrix.L1_row_norm[level], num_sweeps, level);
  //       }
  //    }
  //    else {
  //       if ((all_data->input.solver == MULTADD ||
  //            all_data->input.solver == ASYNC_MULTADD)
  //            &&
  //           (all_data->input.num_post_smooth_sweeps > 0 &&
  //            all_data->input.num_pre_smooth_sweeps > 0)){
  //          SEQ_SymmetricJacobi(all_data, A, f, u, y, r, num_sweeps, level);
  //       }
  //       else {
  //          SEQ_Jacobi(all_data, A, f, u, y, num_sweeps, level);
  //       }
  //    }
  // }
}
