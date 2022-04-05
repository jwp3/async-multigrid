#include "Main.hpp"
#include "SMEM_MatVec.hpp"
#include "Misc.hpp"
#include "SMEM_ExtendedSystem.hpp"

using namespace std;
void ExtendedSystemImplicitMatVec(AllData *all_data);

void SMEM_ExtendedSystemSolve(AllData *all_data)
{
   double sum = 0;
   double r_inner_prod_glob;
   int num_threads = all_data->input.num_threads;
   int num_levels = all_data->grid.num_levels;

   hypre_ParAMGData *amg_data;
   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **P_array;
   hypre_ParCSRMatrix **R_array;
   hypre_ParVector **U_array;
   hypre_ParVector **F_array;
   hypre_ParVector *V;
   HYPRE_Real *v;


   hypre_CSRMatrix *A;
   HYPRE_Real *x;
   HYPRE_Real *b;
   HYPRE_Real *r;
   HYPRE_Real *z;

   HYPRE_Real *A_data;
   HYPRE_Int *A_i;
   HYPRE_Int *A_j;
   int n;
   int chunk_size = 1;//n / num_threads;

   double delta = all_data->cheby.delta;
   int cache_line = 1;
   int *level_converge = (int *)calloc(num_levels * cache_line, sizeof(int));
   int *iters;
   int T;
   if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
       T = num_threads;
   }
   else {
       T = num_levels;
   }
   iters = (int *)calloc(num_threads * cache_line, sizeof(int));
   double *wtime = (double *)calloc(num_threads * cache_line, sizeof(double));
   double *r_norm2_glob = (double *)calloc(num_threads * cache_line, sizeof(double));
   for (int t = 0; t < num_threads; t++){
      iters[t * cache_line] = 1;
      r_norm2_glob[t] = 100; 
   }

   unsigned int *usec_vec = (unsigned int *)calloc(num_threads, sizeof(unsigned int));
   int *sleep_flag_vec = (int *)calloc(num_threads, sizeof(int));
   int num_delayed = (int)ceil((double)num_threads * all_data->input.delay_frac);
   int count_delayed = 0;
   srand(0);
   for (int t = num_threads-1; t >= 0; t--){
      if (count_delayed == num_delayed) break;
      count_delayed++;
      usec_vec[t] = (int)RandDouble(0.0, (double)all_data->input.delay_usec);
      sleep_flag_vec[t] = 1;
   }

   int min_relax, max_relax = 0;
   double mean_wtime = 0, mean_relax = 0, min_wtime, max_wtime = 0;

   int resnorm_converge_flag = 0;
   int glob_done_iters = 0;

   amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
      A_array = hypre_ParAMGDataAArray(amg_data);
      P_array = hypre_ParAMGDataPArray(amg_data);
      R_array = hypre_ParAMGDataRArray(amg_data);
      U_array = hypre_ParAMGDataUArray(amg_data);
      F_array = hypre_ParAMGDataFArray(amg_data);
      V = hypre_ParAMGDataVtemp(amg_data);
      v = hypre_VectorData(hypre_ParVectorLocalVector(V));
   
   
      A = all_data->matrix.AA;
      x = all_data->vector.xx;
      b = all_data->vector.bb;
      r = all_data->vector.rr;
      z = all_data->vector.zz;
   
      A_data = hypre_CSRMatrixData(A);
      A_i = hypre_CSRMatrixI(A);
      A_j = hypre_CSRMatrixJ(A);
      n = hypre_CSRMatrixNumRows(A);
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[0],
                                         U_array[0],
                                         1.0,
                                         F_array[0],
                                         V);
      all_data->output.r0_norm2 = Parfor_Norm2(v, all_data->grid.n[0]);
      SMEM_Sync_Residual(all_data, A, b, x, z, r);
      all_data->output.r0_norm2_ext_sys = Parfor_Norm2(r, n);
      #pragma omp parallel for
      for (int i = 0; i < n; i++){
         x[i] = x[i] + all_data->cheby.delta * r[i] / A_data[A_i[i]];
      }
   }
   else {
      all_data->output.r0_norm2 = Parfor_InnerProd(all_data->vector.f[0], all_data->grid.n[0]);
      all_data->output.r0_norm2_ext_sys = all_data->output.r0_norm2;
      all_data->output.r0_norm2 = sqrt(all_data->output.r0_norm2);
      for (int level = 0; level < num_levels-1; level++){
         int fine_grid = level;
         int coarse_grid = level + 1;
         SMEM_Sync_Parfor_Restrict(all_data,
                                   all_data->matrix.R[fine_grid],
                                   all_data->vector.f[fine_grid],
                                   all_data->vector.f[coarse_grid],
                                   fine_grid, coarse_grid);
         all_data->output.r0_norm2_ext_sys += Parfor_InnerProd(all_data->vector.f[coarse_grid], all_data->grid.n[coarse_grid]);
      }
      all_data->output.r0_norm2_ext_sys = sqrt(all_data->output.r0_norm2_ext_sys);
      for (int level = 0; level < num_levels; level++){
         HYPRE_Int *A_i = hypre_CSRMatrixI(all_data->matrix.A[level]);
         HYPRE_Real *A_data = hypre_CSRMatrixData(all_data->matrix.A[level]);
         for (int i = 0; i < all_data->grid.n[level]; i++){
            all_data->vector.y[level][i] = 0;
            all_data->vector.u[level][i] = delta * all_data->vector.f[level][i] / A_data[A_i[i]];
         }
         SMEM_Sync_Parfor_MatVec(all_data,
                                 all_data->matrix.A[level],
                                 all_data->vector.u[level],
                                 all_data->vector.e[level]);
      }
   }

   double start = omp_get_wtime();
   #pragma omp parallel
   { 
      int tid = omp_get_thread_num();
      int ns, ne, ns_col, ne_col, n_loc, nnz_loc;
      int fine_grid, coarse_grid;
      int finest_level, coarsest_level;
      int thread_level;

      double residual_start;
      double smooth_start;
      double restrict_start;
      double prolong_start;
      double A_matvec_start;
      double vec_start;
      double innerprod_start;

      double r_inner_prod_loc;
      double mu = all_data->cheby.mu;
      double mu22 = pow(2.0 * mu, 2.0);
      double omega = 2.0;

      int i_loc, jj_loc, loc_iters = 1;
      iters[tid * cache_line] = 1; 
      unsigned int usec;
      int sleep_flag;
      
      HYPRE_Real *A_data_loc, *y_loc, *r_loc, **y, **r, **z, *b_loc;
      HYPRE_Int *A_i_loc, *A_j_loc;

      if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
         if (all_data->input.omp_parfor_flag == 1){
            n_loc = 0;
            nnz_loc = 0;
            #pragma omp for nowait schedule(static, chunk_size)
            for (int i = 0; i < n; i++){
               n_loc++;
               nnz_loc += A_i[i+1] - A_i[i] + 1;
            }
         }
         else {
            ns = all_data->thread.AA_NS[tid];
            ne = all_data->thread.AA_NE[tid];
            n_loc = ne - ns + 1;
            for (int i = ns; i < ne; i++){
               nnz_loc += A_i[i+1] - A_i[i] + 1;
            }
         }
         
         y_loc = (HYPRE_Real *)calloc(n_loc, sizeof(HYPRE_Real));
         r_loc = (HYPRE_Real *)calloc(n_loc, sizeof(HYPRE_Real));
         b_loc = (HYPRE_Real *)calloc(n_loc, sizeof(HYPRE_Real));
         A_i_loc = (HYPRE_Int *)calloc(n_loc+1, sizeof(HYPRE_Real));
         A_j_loc = (HYPRE_Int *)calloc(nnz_loc, sizeof(HYPRE_Real));
         A_data_loc = (HYPRE_Real *)calloc(nnz_loc, sizeof(HYPRE_Real));

         if (all_data->input.omp_parfor_flag == 1){
            i_loc = 0;
            jj_loc = 0;
            #pragma omp for nowait schedule(static, chunk_size)
            for (int i = 0; i < n; i++){
               b_loc[i_loc] = b[i];
               A_i_loc[i_loc] = A_i[i];
               A_i_loc[i_loc+1] = A_i[i+1];
               for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
                  A_data_loc[jj_loc] = A_data[jj];
                  A_j_loc[jj_loc] = A_j[jj];
                  jj_loc++;
               }
               i_loc++;
            }
         }
         else {
            i_loc = 0;
            jj_loc = 0;
            for (int i = ns; i < ne; i++){
               b_loc[i_loc] = b[i];
               A_i_loc[i_loc] = A_i[i];
               A_i_loc[i_loc+1] = A_i[i+1];
               for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
                  A_data_loc[jj_loc] = A_data[jj];
                  A_j_loc[jj_loc] = A_j[jj];
                  jj_loc++;
               }
               i_loc++;
            }
         }
      }
      else {
         y = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         r = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         for (int q = 0; q < all_data->thread.thread_levels[tid].size(); q++){
            thread_level = all_data->thread.thread_levels[tid][q];
            // ns = row_ns[thread_level][tid];
            // ne = row_ne[thread_level][tid];
            ns = all_data->thread.A_ns[thread_level][tid];
            ne = all_data->thread.A_ne[thread_level][tid];
            int n_loc = ne - ns + 1;
            int n = all_data->grid.n[thread_level]; 
            y[thread_level] = (HYPRE_Real *)calloc(n_loc, sizeof(HYPRE_Real));
            r[thread_level] = (HYPRE_Real *)calloc(n_loc, sizeof(HYPRE_Real));
         }         
      }

      srand(tid);
      usec = 0;
      sleep_flag = 0;
      int delayed_thread = num_threads-1;
      if (all_data->input.delay_type == DELAY_SOME || all_data->input.delay_type == DELAY_ALL){
         sleep_flag = sleep_flag_vec[tid];
         usec = usec_vec[tid];
      }
      else if (tid == delayed_thread && all_data->input.delay_type == DELAY_ONE){
         sleep_flag = 1;
         usec = all_data->input.delay_usec;
      }
      else if (tid == delayed_thread && all_data->input.delay_type == FAIL_ONE){
         usec = all_data->input.delay_usec;
      }

      #pragma omp barrier
      double start = omp_get_wtime();
      if (all_data->input.num_cycles > 1)
      while(1){
         if (tid == delayed_thread && all_data->input.delay_type == FAIL_ONE){
            if (loc_iters == all_data->input.fail_iter){
               sleep_flag = 1;
            }
            else {
               sleep_flag = 0;
            }
         }

         if (sleep_flag == 1){
            usec = (int)RandDouble(0.0, 2.0 * (double)all_data->input.delay_usec);
#ifdef WINDOWS
            this_thread::sleep_for(chrono::microseconds(usec));
#else
            usleep(usec);
#endif
         }

         if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){  
            A_matvec_start = omp_get_wtime();
            i_loc = 0;
            if (all_data->input.omp_parfor_flag == 1){
               #pragma omp for schedule(static, chunk_size) nowait
               for (int i = 0; i < n; i++){
                  r_loc[i_loc] = b_loc[i_loc];
                  for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
                     r_loc[i_loc] -= A_data[jj] * x[A_j[jj]];
                  }
                  i_loc++;
               } 
            }
            else {
               for (int i = ns; i < ne; i++){
                  r_loc[i_loc] = b_loc[i_loc];
                  for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
                     r_loc[i_loc] -= A_data[jj] * x[A_j[jj]];
                  }
                  i_loc++;
               }
            }
            all_data->output.A_matvec_wtime[tid] += omp_get_wtime() - A_matvec_start;

            if (all_data->input.async_flag == 0){
               #pragma omp barrier
            }

            vec_start = omp_get_wtime();
            i_loc = 0;
            if (all_data->input.omp_parfor_flag == 1){
               if (all_data->input.check_resnorm_flag == 1 && loc_iters > 1){
                  r_inner_prod_loc = 0;
                  #pragma omp for schedule(static, chunk_size) nowait
                  for (int i = 0; i < n; i++){
                     double x_tmp = x[i];
                     double x_write = y_loc[i_loc] + omega * (delta * r_loc[i_loc] / A_data[A_i[i]] + x_tmp - y_loc[i_loc]);
                     //#pragma omp atomic write
                     x[i] = x_write;
                     y_loc[i_loc] = x_tmp;
                     r_inner_prod_loc += pow(r_loc[i_loc], 2.0);
                     i_loc++;
                  }
                  r_norm2_glob[tid * cache_line] = r_inner_prod_loc;
               }
               else {
                  #pragma omp for schedule(static, chunk_size) nowait
                  for (int i = 0; i < n; i++){
                     double x_tmp = x[i];
                     double x_write = y_loc[i_loc] + omega * (delta * r_loc[i_loc] / A_data[A_i[i]] + x_tmp - y_loc[i_loc]);
                     //#pragma omp atomic write
                     x[i] = x_write;
                     y_loc[i_loc] = x_tmp;
                     i_loc++;
                  }
               }
            }
            else {
               if (all_data->input.check_resnorm_flag == 1 && loc_iters > 1){
                  r_inner_prod_loc = 0;
                  for (int i = ns; i < ne; i++){
                     double x_tmp = x[i];
                     double x_write = y_loc[i_loc] + omega * (delta * r_loc[i_loc] / A_data[A_i[i]] + x_tmp - y_loc[i_loc]);
                     //#pragma omp atomic write
                     x[i] = x_write;
                     y_loc[i_loc] = x_tmp;
                     r_inner_prod_loc += pow(r_loc[i_loc], 2.0);
                     i_loc++;
                  }
                  r_norm2_glob[tid * cache_line] = r_inner_prod_loc;
               }
               else {
                  for (int i = ns; i < ne; i++){
                     double x_tmp = x[i];
                     double x_write = y_loc[i_loc] + omega * (delta * r_loc[i_loc] / A_data[A_i[i]] + x_tmp - y_loc[i_loc]);
                     //#pragma omp atomic write
                     x[i] = x_write;
                     y_loc[i_loc] = x_tmp;
                     i_loc++;
                  }
               }
            }
            all_data->output.vec_wtime[tid] += omp_get_wtime() - vec_start;

            if (all_data->input.async_flag == 0){
               #pragma omp barrier
            }
         }
         else {
            HYPRE_Real **u = all_data->vector.u;
            HYPRE_Real **f = all_data->vector.f;
            HYPRE_Real **e = all_data->vector.e;
            HYPRE_Real **z = all_data->vector.z;
            HYPRE_Real **u_smooth = all_data->vector.u_smooth;
            //HYPRE_Real **y = all_data->vector.y;
            //HYPRE_Real **r = all_data->vector.r;

            hypre_CSRMatrix **P = all_data->matrix.P;
            hypre_CSRMatrix **R = all_data->matrix.R;
            hypre_CSRMatrix **A = all_data->matrix.A;
            int **P_ns = all_data->thread.P_ns;
            int **P_ne = all_data->thread.P_ne;
            int **R_ns = all_data->thread.R_ns;
            int **R_ne = all_data->thread.R_ne;
            int **A_ns = all_data->thread.A_ns;
            int **A_ne = all_data->thread.A_ne;
            int **row_ns = all_data->thread.row_ns;
            int **row_ne = all_data->thread.row_ne;

            r_inner_prod_loc = 0;
            for (int q = 0; q < all_data->thread.thread_levels[tid].size(); q++){
               thread_level = all_data->thread.thread_levels[tid][q];
               HYPRE_Real **z1 = all_data->level_vector[thread_level].z1;
               HYPRE_Real **z2 = all_data->level_vector[thread_level].z2;

               coarsest_level = num_levels-1;
               ns = all_data->thread.A_ns[coarsest_level][tid];
               ne = all_data->thread.A_ne[coarsest_level][tid];
               vec_start = omp_get_wtime();
               for (int i = ns; i < ne; i++){
                  z1[coarsest_level][i] = u[coarsest_level][i];
               }
               all_data->output.vec_wtime[tid] += omp_get_wtime() - vec_start;
               for (int level = coarsest_level-1; level >= thread_level; level--){
                  fine_grid = level;
                  coarse_grid = level + 1;
                  ns = P_ns[fine_grid][tid];
                  ne = P_ne[fine_grid][tid];
                  SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
                  prolong_start = omp_get_wtime();
                  SMEM_MatVec(all_data,
                              P[fine_grid],
                              z1[coarse_grid],
                              z1[fine_grid],
                              ns, ne);
                  all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
                  vec_start = omp_get_wtime();
                  for (int i = ns; i < ne; i++){
                     double ui_read;
                    // #pragma omp atomic read
                     ui_read = u[fine_grid][i];
                     z1[fine_grid][i] += ui_read;
                  }
                  all_data->output.vec_wtime[tid] += omp_get_wtime() - vec_start;
               }

               finest_level = 0;
               ns = row_ns[finest_level][tid];
               ne = row_ne[finest_level][tid];
               for (int level = finest_level; level < thread_level; level++){
                  fine_grid = level;
                  coarse_grid = level + 1;
                  vec_start = omp_get_wtime();
                  if (level == finest_level){
                     for (int i = ns; i < ne; i++){
                       // #pragma omp atomic read
                        z2[fine_grid][i] = e[fine_grid][i];
                     }
                  }
                  else {
                     for (int i = ns; i < ne; i++){
                        double ei_read;
                       // #pragma omp atomic read
                        ei_read = e[fine_grid][i];
                        z2[fine_grid][i] += ei_read;
                     }
                  }
                  all_data->output.vec_wtime[tid] += omp_get_wtime() - vec_start;
                  SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
                  ns = R_ns[fine_grid][tid];
                  ne = R_ne[fine_grid][tid];
                  ns_col = row_ns[coarse_grid][tid];
                  ne_col = row_ne[coarse_grid][tid];
                  int shift_t = tid - all_data->thread.level_threads[thread_level][0];
                  restrict_start = omp_get_wtime();
                  SMEM_Restrict(all_data,
                                R[fine_grid],
                                z2[fine_grid],
                                z2[coarse_grid],
                                fine_grid, coarse_grid,
                                ns, ne, ns_col, ne_col,
                                shift_t, thread_level); 
                  all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
                  if (all_data->input.construct_R_flag == 1){
                     ns = R_ns[fine_grid][tid];
                     ne = R_ne[fine_grid][tid];
                  }
                  else {
                     ns = row_ns[coarse_grid][tid];
                     ne = row_ne[coarse_grid][tid];
                  }
               }

               if (all_data->input.async_flag == 1){
                  SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               }
            }

            if (all_data->input.async_flag == 0){
               #pragma omp barrier
            }


            for (int q = 0; q < all_data->thread.thread_levels[tid].size(); q++){ 
               thread_level = all_data->thread.thread_levels[tid][q];
               HYPRE_Real **z1 = all_data->level_vector[thread_level].z1;
               HYPRE_Real **z2 = all_data->level_vector[thread_level].z2;

               ns = A_ns[thread_level][tid];
               ne = A_ne[thread_level][tid];
               A_matvec_start = omp_get_wtime();
               SMEM_MatVec(all_data,
                           A[thread_level],
                           z1[thread_level],
                           z[thread_level],
                           ns, ne);
               all_data->output.A_matvec_wtime[tid] += omp_get_wtime() - A_matvec_start;

               HYPRE_Int *A_i = hypre_CSRMatrixI(all_data->matrix.A[thread_level]);
               HYPRE_Int *A_j = hypre_CSRMatrixJ(all_data->matrix.A[thread_level]);
               HYPRE_Real *A_data = hypre_CSRMatrixData(all_data->matrix.A[thread_level]);

               double *diag_scale;
               if (all_data->input.smoother == L1_JACOBI ||
                   all_data->input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL){
                  diag_scale = hypre_ParAMGDataL1Norms(amg_data)[thread_level];
                  //diag_scale = all_data->matrix.L1_row_norm[thread_level];
               }
               else {
                  diag_scale = all_data->matrix.A_diag[thread_level];
               }

               vec_start = omp_get_wtime();
               ////if (thread_level == coarsest_level){
               ////   for (int i = ns; i < ne; i++){
               ////      int ii = i-ns;
               ////      z[thread_level][i] += z2[thread_level][i];
               ////      r[thread_level][ii] = f[thread_level][i] - z[thread_level][i];
               ////      hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[coarsest_level]))[i] = r[thread_level][ii];
               ////   }
               ////   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               ////   if (tid == num_threads-1){
               ////      hypre_GaussElimSolve(amg_data, coarsest_level, 9);
               ////   }
               ////   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               ////   for (int i = ns; i < ne; i++){
               ////      int ii = i-ns;
               ////      double u_prev = u[thread_level][i];
               ////      double ui = y[thread_level][ii] + omega * (delta * hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataUArray(amg_data)[coarsest_level]))[i] + u[thread_level][i] - y[thread_level][ii]);
               ////     // #pragma omp atomic write
               ////      u[thread_level][i] = ui;
               ////      y[thread_level][ii] = u_prev;
               ////   }
               ////}
               ////else {
               //   for (int i = ns; i < ne; i++){
               //      //double Axi = 0.0;
               //      //for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               //      //   Axi += A_data[jj] * z1[thread_level][A_j[jj]];
               //      //}
               //      //z[thread_level][i] = Axi + z2[thread_level][i];

               //      int ii = i-ns; 
               //      z[thread_level][i] += z2[thread_level][i];
               //      r[thread_level][ii] = f[thread_level][i] - z[thread_level][i];
               //      double u_prev = u[thread_level][i];
               //      double ui = y[thread_level][ii] + omega * (delta * r[thread_level][ii] / diag_scale[i] + u[thread_level][i] - y[thread_level][ii]);
               //     // #pragma omp atomic write
               //      u[thread_level][i] = ui;
               //      y[thread_level][ii] = u_prev;
               //   }
               ////}

               if (all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL ||
                   all_data->input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL){
                  for (int i = ns; i < ne; i++){
                     u_smooth[thread_level][i] = 0;
                     int ii = i-ns;
                     z[thread_level][i] += z2[thread_level][i];
                     r[thread_level][ii] = f[thread_level][i] - z[thread_level][i];
                  }
                  for (int i = ns; i < ne; i++){
                     int ii = i-ns;
                     double res = r[thread_level][ii];
                     for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
                        int ii = A_j[jj];
                        //if (ii >= ns && ii < ne){
                           res -= A_data[jj] * u_smooth[thread_level][ii];
                        //}
                        //else {
                        //   res -= A_data[jj] * u_smooth_prev[ii];
                        //}
                     }
                     u_smooth[thread_level][i] += res / diag_scale[i];
                  }

                  //SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               }
               else {
                  for (int i = ns; i < ne; i++){
                     int ii = i-ns;
                     z[thread_level][i] += z2[thread_level][i];
                     r[thread_level][ii] = f[thread_level][i] - z[thread_level][i];
                     u_smooth[thread_level][i] = r[thread_level][ii] / diag_scale[i];
                  }
               }

               for (int i = ns; i < ne; i++){
                  int ii = i-ns;
                  double u_prev = u[thread_level][i];
                  double ui = y[thread_level][ii] + omega * (delta * u_smooth[thread_level][i] + u[thread_level][i] - y[thread_level][ii]);
                  // #pragma omp atomic write
                  u[thread_level][i] = ui;
                  y[thread_level][ii] = u_prev;
               }

               all_data->output.vec_wtime[tid] += omp_get_wtime() - vec_start;

               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);

               ns = A_ns[thread_level][tid];
               ne = A_ne[thread_level][tid];
               A_matvec_start = omp_get_wtime();
               SMEM_MatVec(all_data,
                           A[thread_level],
                           u[thread_level],
                           z[thread_level],
                           ns, ne);
               all_data->output.A_matvec_wtime[tid] += omp_get_wtime() - A_matvec_start;
               vec_start = omp_get_wtime();
               for (int i = ns; i < ne; i++){
                 // #pragma omp atomic write
                  e[thread_level][i] = z[thread_level][i];
               }
               all_data->output.vec_wtime[tid] += omp_get_wtime() - vec_start;
               if (all_data->input.check_resnorm_flag == 1 && loc_iters > 1){
                 // ns = row_ns[thread_level][tid];
                 // ne = row_ne[thread_level][tid];
                  ns = A_ns[thread_level][tid];
                  ne = A_ne[thread_level][tid];
                  innerprod_start = omp_get_wtime();
                  for (int i = ns; i < ne; i++){
                     int ii = i-ns;
                     r_inner_prod_loc += pow(r[thread_level][ii], 2.0);
                  }
                  r_norm2_glob[tid * cache_line] = r_inner_prod_loc;
                  all_data->output.innerprod_wtime[tid] += omp_get_wtime() - innerprod_start;
                  all_data->output.vec_wtime[tid] += omp_get_wtime() - innerprod_start;
               }
            }
         }

         omega = 1.0 / (1.0 - omega / mu22);
         loc_iters++; 

         if (all_data->input.async_flag == 0){
            if (loc_iters == all_data->input.num_cycles){
               break;
            }
            if (all_data->input.check_resnorm_flag == 1 && loc_iters > 1){
               if (tid == 0){
                  innerprod_start = omp_get_wtime();
                  r_inner_prod_glob = 0;
                  for (int t = 0; t < num_threads; t++){
                     r_inner_prod_glob += r_norm2_glob[t * cache_line];
                  }
                  all_data->output.innerprod_wtime[tid] += omp_get_wtime() - innerprod_start;
               }
               #pragma omp barrier
               if (sqrt(r_inner_prod_glob)/all_data->output.r0_norm2_ext_sys < all_data->input.tol ||
                   loc_iters == all_data->input.num_cycles){
                  break;
               }
            }
            else {
               #pragma omp barrier
            }
         }
         else {
            if (all_data->input.check_resnorm_flag == 1 && loc_iters > 1){
               if (tid == 0){
                  innerprod_start = omp_get_wtime();
                  double r_inner_prod = 0;
                  for (int t = 0; t < num_threads; t++){
                     r_inner_prod += r_norm2_glob[t * cache_line];
                  }
                  all_data->output.innerprod_wtime[tid] += omp_get_wtime() - innerprod_start;
                  if (sqrt(r_inner_prod)/all_data->output.r0_norm2_ext_sys < all_data->input.tol){
                     resnorm_converge_flag = 1;
                  }
               }
            }
        
            if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
               if (resnorm_converge_flag == 1){
                  break;
               }
               if (loc_iters >= all_data->input.num_cycles){
                  if (loc_iters == all_data->input.num_cycles){
                     #pragma omp atomic
                     glob_done_iters++;
                  }
                  if (glob_done_iters == num_threads){
                     break;
                  }
               }
            }
            else {
               thread_level = all_data->thread.thread_levels[tid][0];
               if (tid == all_data->thread.barrier_root[thread_level]){  
                  if (loc_iters == all_data->input.num_cycles){
                     for (int q = 0; q < all_data->thread.thread_levels[tid].size(); q++){
                        #pragma omp atomic
                        glob_done_iters++;
                     }
                  }
                  if (resnorm_converge_flag == 1 || glob_done_iters == num_levels){
                     level_converge[thread_level] = 1;
                  } 
               }
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               if (level_converge[thread_level] == 1){
                  break;
               }
            }
         }
      }
      all_data->grid.local_num_correct[tid] = loc_iters;

      if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
         free(y_loc);
         free(r_loc);
         free(b_loc);
         free(A_i_loc);
         free(A_j_loc);
         free(A_data_loc);
      }
      else {
         for (int q = 0; q < all_data->thread.thread_levels[tid].size(); q++){
            thread_level = all_data->thread.thread_levels[tid][q];
            free(y[thread_level]);
            free(r[thread_level]);
         }
         free(y);
         free(r);
      }
   }
   all_data->output.solve_wtime = omp_get_wtime() - start;

   if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
      SMEM_Sync_Residual(all_data, A, b, x, z, r);
      all_data->output.r_norm2_ext_sys = Parfor_Norm2(r, n);
   
      HYPRE_Real *u, *u_f, *u_c, *f;
      int *disp = all_data->grid.disp;
      for (int i = 0; i < all_data->grid.n[0]; i++){
         v[i] = 0.0;
      }
      for (int level = 0; level < num_levels; level++){
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
     // for (int i = 0; i < n; i++){
     //    printf("%e %e\n", x[i], r[i]);
     // }
      u = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
      f = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));
      //for (int i = 0; i < all_data->grid.n[0]; i++){
      //   printf("%e %e\n", u[i], f[i]);
      //}
      hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                         A_array[0],
                                         U_array[0],
                                         1.0,
                                         F_array[0],
                                         V);
      all_data->output.r_norm2 = Parfor_Norm2(v, all_data->grid.n[0]);
   }
   else {
      ExtendedSystemImplicitMatVec(all_data);
      HYPRE_Real **r = all_data->vector.r;
      HYPRE_Real **f = all_data->vector.f;
      HYPRE_Real **z = all_data->vector.z;
      double r_inner_prod = 0;
      for (int level = 0; level < num_levels; level++){
          HYPRE_Real **z2 = all_data->level_vector[level].z2;
          for (int i = 0; i < all_data->grid.n[level]; i++){
             z[level][i] += z2[level][i];
             r[level][i] = f[level][i] - z[level][i];
             r_inner_prod += pow(r[level][i], 2.0);
          }
      }
      all_data->output.r_norm2_ext_sys = sqrt(r_inner_prod);
      
 
      for (int level = all_data->grid.num_levels-2; level >= 0; level--){
         int fine_grid = level;
         int coarse_grid = level + 1;
         SMEM_Sync_Parfor_MatVec(all_data,
                                 all_data->matrix.P[fine_grid],
                                 all_data->vector.u[coarse_grid],
                                 all_data->vector.e[fine_grid]);
         for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
            all_data->vector.u[fine_grid][i] += all_data->vector.e[fine_grid][i];
         }
      }
      //for (int i = 0; i < all_data->grid.n[0]; i++){
      //   printf("%e %e\n", all_data->vector.u[0][i], all_data->vector.f[0][i]);
      //}
      SMEM_Sync_Parfor_Residual(all_data,
                                all_data->matrix.A[0],
                                all_data->vector.f[0],
                                all_data->vector.u[0],
                                all_data->vector.y[0],
                                all_data->vector.r[0]);
      all_data->output.r_norm2 = Parfor_Norm2(all_data->vector.r[0], all_data->grid.n[0]);
   }

   free(level_converge);
   free(iters);
   free(wtime);
   free(r_norm2_glob);
   free(usec_vec);
   free(sleep_flag_vec);

  // printf("ext. sys. rel. res. norm %e, rel. res. norm %e, time %e\n",
  // printf("%e %e %e %e %e %f %d %d\n",
  //        all_data->output.r_norm2_ext_sys/all_data->output.r0_norm2_ext_sys,
  //        all_data->output.r_norm2/all_data->output.r0_norm2,
  //        mean_wtime, min_wtime, max_wtime,
  //        mean_relax, min_relax, max_relax);

  // for (int i = 0; i < n; i++){
  //    printf("%d %e\n", i, x[i]);
  // }
  // printf("c = %e, c_prev = %e, omega = %e, mu = %e, delta = %e\n",
  //        c[0], c_prev[0], omega[0], mu, delta);
}

void ExtendedSystemImplicitMatVec(AllData *all_data)
{
   int num_levels = all_data->grid.num_levels;
   #pragma omp parallel
   {
      for (int thread_level = 0; thread_level < num_levels; thread_level++){
         int fine_grid, coarse_grid;
         int finest_level, coarsest_level;

         HYPRE_Real **u = all_data->vector.u;
         HYPRE_Real **f = all_data->vector.f;
         HYPRE_Real **e = all_data->vector.e;
         HYPRE_Real **y = all_data->vector.y;
         HYPRE_Real **r = all_data->vector.r;
         HYPRE_Real **z = all_data->vector.z;
         HYPRE_Real **u_prev = all_data->vector.u_prev;
         HYPRE_Real **z1 = all_data->level_vector[thread_level].z1;
         HYPRE_Real **z2 = all_data->level_vector[thread_level].z2;
         hypre_CSRMatrix **P = all_data->matrix.P;
         hypre_CSRMatrix **R = all_data->matrix.R;
         hypre_CSRMatrix **A = all_data->matrix.A;
         int **P_ns = all_data->thread.P_ns;
         int **P_ne = all_data->thread.P_ne;
         int **R_ns = all_data->thread.R_ns;
         int **R_ne = all_data->thread.R_ne;
         int **A_ns = all_data->thread.A_ns;
         int **A_ne = all_data->thread.A_ne;

         coarsest_level = num_levels-1;
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[coarsest_level]; i++){
            z1[coarsest_level][i] = u[coarsest_level][i];
         }
         for (int level = coarsest_level-1; level >= thread_level; level--){
            fine_grid = level;
            coarse_grid = level + 1;
            SMEM_Sync_Parfor_MatVec(all_data,
                                    P[fine_grid],
                                    z1[coarse_grid],
                                    z1[fine_grid]);
            #pragma omp for
            for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
               z1[fine_grid][i] += u[fine_grid][i];
            }
         }
         SMEM_Sync_Parfor_MatVec(all_data,
                                 A[thread_level],
                                 z1[thread_level],
                                 z[thread_level]);
         finest_level = 0;
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[finest_level]; i++){
            z2[finest_level][i] = 0.0;
         }
         for (int level = finest_level; level < thread_level; level++){
            fine_grid = level;
            coarse_grid = level + 1;
            #pragma omp for
            for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
               z2[fine_grid][i] += e[fine_grid][i];
            }
            SMEM_Sync_Parfor_Restrict(all_data,
                                      R[fine_grid],
                                      z2[fine_grid],
                                      z2[coarse_grid],
                                      fine_grid, coarse_grid);
         }
      }
   }
}
