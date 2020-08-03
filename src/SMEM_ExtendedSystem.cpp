#include "Main.hpp"
#include "SMEM_MatVec.hpp"
#include "Misc.hpp"
#include "SMEM_ExtendedSystem.hpp"

using namespace std;

void SMEM_ExtendedSystemSolve(AllData *all_data)
{
   double sum = 0;
   int num_threads = all_data->input.num_threads;
   int cache_line = 1;
   int *iters = (int *)calloc(num_threads * cache_line, sizeof(int));
   double *wtime = (double *)calloc(num_threads * cache_line, sizeof(double));
   double *r_norm2_glob = (double *)calloc(num_threads * cache_line, sizeof(double));
   for (int t = 0; t < num_threads; t++){
      iters[t * cache_line] = 1;
   }

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector *V = hypre_ParAMGDataVtemp(amg_data);
   HYPRE_Real *v = hypre_VectorData(hypre_ParVectorLocalVector(V));
   

   hypre_CSRMatrix *A = all_data->matrix.AA;
   HYPRE_Real *x = all_data->vector.xx;
   HYPRE_Real *b = all_data->vector.bb;
   HYPRE_Real *r = all_data->vector.rr;
   HYPRE_Real *z = all_data->vector.zz;
  // HYPRE_Real *y = all_data->vector.yy;
  // HYPRE_Real *x_prev = all_data->vector.xx_prev;

  // double *omega = all_data->cheby.omega;
  // double *c = all_data->cheby.c;
  // double *c_prev = all_data->cheby.c_prev;

   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   int n = hypre_CSRMatrixNumRows(A);
   int chunk_size = 1;//n / num_threads;

   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                      A_array[0],
                                      U_array[0],
                                      1.0,
                                      F_array[0],
                                      V);
   all_data->output.r0_norm2 = Parfor_Norm2(v, all_data->grid.n[0]);

   SMEM_Sync_Parfor_Residual(all_data, A, b, x, z, r);
   all_data->output.r0_norm2_ext_sys = Parfor_Norm2(r, n);
   #pragma omp parallel for
   for (int i = 0; i < n; i++){
      x[i] = x[i] + all_data->cheby.delta * r[i] / A_data[A_i[i]];
   }

   unsigned int *usec_vec = (unsigned int *)calloc(num_threads, sizeof(unsigned int));
   srand(0);
   int num_delayed = (int)ceil((double)num_threads * all_data->input.delay_frac);
   int count_delayed = 0;
   for (int t = num_threads-1; t >= 0; t--){
      if (count_delayed == num_delayed) break;
      count_delayed++;
      usec_vec[t] = all_data->input.delay_usec;
   }

   int min_relax, max_relax = 0;
   double mean_wtime = 0, mean_relax = 0, min_wtime, max_wtime = 0;

   int converge_flag = 0;
   int glob_done_iters = 0;
  // double start = omp_get_wtime();
   #pragma omp parallel
   { 
      int tid = omp_get_thread_num();
      int *loc_iters = (int *)calloc(num_threads, sizeof(int));
      int ns, ne, n_loc;
      ne = all_data->thread.AA_NE[tid];
      if (all_data->input.omp_parfor_flag == 1){
         ns = 0;
         n_loc = n;
      }
      else {
         ns = all_data->thread.AA_NS[tid];
         n_loc = ne - ns + 1;
      }
      
      HYPRE_Real *y_loc = (HYPRE_Real *)calloc(n_loc, sizeof(HYPRE_Real));
      HYPRE_Real *r_loc = (HYPRE_Real *)calloc(n_loc, sizeof(HYPRE_Real));

     // double omega = all_data->cheby.omega[tid];
      double mu = all_data->cheby.mu;
      double c_prev = 1.0;
      double c = mu;
      double delta = all_data->cheby.delta;
      double mu22 = pow(2.0 * mu, 2.0);
      double omega = 2.0 * mu * c_prev / c;
      
     // vector <int> col_read;
      #pragma omp for schedule(static, chunk_size)
      for (int i = 0; i < n; i++){
     // for (int i = ns; i < ne; i++){
         r_loc[i-ns] = b[i];
         for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
            r_loc[i-ns] -= A_data[jj] * x[A_j[jj]];
           // col_read.push_back(A_j[jj]);
         }
      }
     // sort(col_read.begin(), col_read.end());
     // col_read.erase(unique(col_read.begin(), col_read.end()), col_read.end());
     // int min_col = *min_element(col_read.begin(), col_read.end());
     // int max_col = *max_element(col_read.begin(), col_read.end());
     // vector<double> x_read(max_col - min_col + 1);

      double r_norm2_squ_loc = 0;
      #pragma omp for schedule(static, chunk_size)
      for (int i = 0; i < n; i++){
     // for (int i = ns; i < ne; i++){
         r_norm2_squ_loc += pow(r_loc[i-ns], 2.0);
      }
      r_norm2_glob[tid * cache_line] = r_norm2_squ_loc;
      int tid_iters = 1;

      srand(tid);
      unsigned int usec = 0;
      int delay_flag = 0;      
      if (all_data->input.delay_flag == DELAY_SOME || all_data->input.delay_flag == DELAY_ALL){
         delay_flag = 1;
         usec = usec_vec[tid];
      }
      else if (tid == num_threads/2 && all_data->input.delay_flag == DELAY_ONE){
         delay_flag = 1;
         usec = all_data->input.delay_usec;
      }

      #pragma omp barrier
      double start = omp_get_wtime();
      while(1){
         if (all_data->input.async_flag == 0){
            #pragma omp for schedule(static, chunk_size)
            for (int i = 0; i < n; i++){
           // for (int i = ns; i < ne; i++){
               double x_tmp = x[i]; 
               x[i] = y_loc[i-ns] + omega * (delta * r_loc[i-ns] / A_data[A_i[i]] + x[i] - y_loc[i-ns]);
               y_loc[i-ns] = x_tmp;
            }
         }
         else { 
           #pragma omp for nowait schedule(static, chunk_size)
           for (int i = 0; i < n; i++){
          //  for (int i = ns; i < ne; i++){
               double x_tmp = x[i];
               double x_write = y_loc[i-ns] + omega * (delta * r_loc[i-ns] / A_data[A_i[i]] + x[i] - y_loc[i-ns]);
              // #pragma omp atomic write
               x[i] = x_write;
               y_loc[i-ns] = x_tmp;
            }
         }

         omega = 1.0 / (1.0 - omega / mu22);
          
        // if (all_data->input.async_flag == 0){
        //    #pragma omp barrier
        // }

        // for (int i = 0; i < col_read.size(); i++){
        //    int j = col_read[i];
        //    x_read[j-min_col] = x[j];
        // }
          
         #pragma omp for nowait schedule(static, chunk_size)
         for (int i = 0; i < n; i++){
       // for (int i = ns; i < ne; i++){
            r_loc[i-ns] = b[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               r_loc[i-ns] -= A_data[jj] * x[A_j[jj]];
              // r_loc[i-ns] -= A_data[jj] * x_read[A_j[jj]-min_col];
            }
         }

         if (all_data->input.check_resnorm_flag == 1){
            double r_norm2_squ_loc = 0;
            if (all_data->input.async_flag == 0){
               #pragma omp for schedule(static, chunk_size)
               for (int i = 0; i < n; i++){
              // for (int i = ns; i < ne; i++){
                  r_norm2_squ_loc += pow(r_loc[i-ns], 2.0);
               }
            }
            else {
               #pragma omp for nowait schedule(static, chunk_size)
               for (int i = 0; i < n; i++){
              // for (int i = ns; i < ne; i++){
                  r_norm2_squ_loc += pow(r_loc[i-ns], 2.0);
               }
            }
            r_norm2_glob[tid * cache_line] = r_norm2_squ_loc;
         }

         if (all_data->input.async_flag == 0){
            tid_iters++;
            #pragma omp barrier
         }
         else {
            iters[tid * cache_line]++;
         }

         if (delay_flag == 1){
            usleep(usec);
         }

         if (all_data->input.async_flag == 0){
            if (tid_iters == all_data->input.num_cycles){
               break;
            }
            if (all_data->input.check_resnorm_flag == 1){
               double r_norm2_squ = 0;
               for (int t = 0; t < num_threads; t++){
                  r_norm2_squ += r_norm2_glob[t * cache_line];
               }
               if (sqrt(r_norm2_squ)/all_data->output.r0_norm2_ext_sys < all_data->input.tol ||
                   tid_iters == all_data->input.num_cycles){
                  break;
               }
            }
         }
         else {
            if (all_data->input.check_resnorm_flag == 1){
               if (tid == 0){
                  double r_norm2_squ = 0;
                  for (int t = 0; t < num_threads; t++){
                     r_norm2_squ += r_norm2_glob[t * cache_line];
                  }
                  if (sqrt(r_norm2_squ)/all_data->output.r0_norm2_ext_sys < all_data->input.tol){
                     converge_flag = 1;
                     break;
                  }

                 // int min_iter = iters[0];
                 // for (int t = 1; t < num_threads; t++){
                 //    int t_iter = iters[t * cache_line];
                 //    if (min_iter > t_iter){
                 //       min_iter = t_iter;
                 //    }
                 // }
                 // if (min_iter >= all_data->input.num_cycles){
                 //    converge_flag = 1;
                 //    break;
                 // }
               }
               else {
                  if (converge_flag == 1){
                     break;
                  }
               }
            }
            
            tid_iters = iters[tid * cache_line];
            if (tid_iters >= all_data->input.num_cycles){
               if (tid_iters == all_data->input.num_cycles){
                  #pragma omp atomic
                  glob_done_iters++;
               }
               if (glob_done_iters == num_threads){
                  break;
               }
            }
         }

         
      }
      wtime[tid] = omp_get_wtime() - start;
      #pragma omp barrier

      for (int t = 0; t < num_threads; t++){
         if (tid == t){
            printf("%d %d %d\n", tid, sched_getcpu(), thread::hardware_concurrency());
         }
         #pragma omp barrier
      }

      if (tid == 0){
         min_relax = iters[0];
         min_wtime = wtime[0];
         for (int t = 0; t < num_threads; t++){
            mean_relax += (double)iters[t * cache_line];
            if (iters[t * cache_line] <= min_relax){
               min_relax = iters[t * cache_line];
            }
            if (iters[t * cache_line] >= max_relax){
               max_relax = iters[t * cache_line];
            }

            mean_wtime += wtime[t];
            if (wtime[t] <= min_wtime){
               min_wtime = wtime[t];
            }
            if (wtime[t] >= max_wtime){
               max_wtime = wtime[t];
            }
         }
         mean_relax /= (double)num_threads;
         mean_wtime /= (double)num_threads;

         if (all_data->input.async_flag == 0){
           mean_relax = min_relax = max_relax = tid_iters;
         }
      }
   }
  // all_data->output.solve_wtime = omp_get_wtime() - start;

   SMEM_Sync_Parfor_Residual(all_data, A, b, x, z, r);
   all_data->output.r_norm2_ext_sys = Parfor_Norm2(r, n);

   HYPRE_Real *u, *u_f, *u_c;
   int *disp = all_data->grid.disp;
   for (int i = 0; i < all_data->grid.n[0]; i++){
      v[i] = 0.0;
   }
   for (int level = 0; level < all_data->grid.num_levels; level++){
      u = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));
      for (int i = 0; i < all_data->grid.n[level]; i++){
         u[i] = x[disp[level]+i];
      }
      for (int inner_level = level-1; inner_level >= 0; inner_level--){
         int fine_grid = inner_level;
         int coarse_grid = inner_level + 1;

         hypre_ParCSRMatrixMatvec(1.0,
                                  P_array[fine_grid],
                                  U_array[coarse_grid],
                                  0.0,
                                  U_array[fine_grid]);
      }
      u = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
      for (int i = 0; i < all_data->grid.n[0]; i++){
         v[i] += u[i];
      }
   }
   hypre_ParVectorCopy(V, U_array[0]);
   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                      A_array[0],
                                      U_array[0],
                                      1.0,
                                      F_array[0],
                                      V);
   all_data->output.r_norm2 = Parfor_Norm2(v, all_data->grid.n[0]);

  // printf("ext. sys. rel. res. norm %e, rel. res. norm %e, time %e\n",
   printf("%e %e %e %e %e %f %d %d\n",
          all_data->output.r_norm2_ext_sys/all_data->output.r0_norm2_ext_sys,
          all_data->output.r_norm2/all_data->output.r0_norm2,
          mean_wtime, min_wtime, max_wtime,
          mean_relax, min_relax, max_relax);

  // for (int i = 0; i < n; i++){
  //    printf("%d %e\n", i, x[i]);
  // }
  // printf("c = %e, c_prev = %e, omega = %e, mu = %e, delta = %e\n",
  //        c[0], c_prev[0], omega[0], mu, delta);
}
