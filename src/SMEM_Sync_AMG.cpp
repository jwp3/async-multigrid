#include "Main.hpp"
#include "SMEM_MatVec.hpp"
#include "SMEM_Smooth.hpp"
#include "Misc.hpp"
#include "SMEM_Solve.hpp"

void SMEM_Sync_Parfor_Vcycle(AllData *all_data)
{
   #pragma omp parallel
   {
      int fine_grid, coarse_grid;
      int tid = omp_get_thread_num();

      double residual_start;
      double smooth_start;
      double restrict_start;
      double prolong_start;

      for (int level = 0; level < all_data->grid.num_levels-1; level++){
         if (level == 0){
            all_data->grid.zero_flags[level] = 0;
         }
         else {
            all_data->grid.zero_flags[level] = 1;
         }
         fine_grid = level;
         coarse_grid = level + 1;
         smooth_start = omp_get_wtime();
         SMEM_Smooth(all_data,
                     all_data->matrix.A[fine_grid],
                     all_data->vector.f[fine_grid],
                     all_data->vector.u[fine_grid],
                     all_data->vector.u_prev[fine_grid],
                     all_data->vector.y[fine_grid],
                     all_data->input.num_pre_smooth_sweeps,
                     fine_grid,
                     0, 0);
         all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
         residual_start = omp_get_wtime();
         SMEM_Sync_Parfor_Residual(all_data,
                                   all_data->matrix.A[fine_grid],
                                   all_data->vector.f[fine_grid],
                                   all_data->vector.u[fine_grid],
                                   all_data->vector.y[fine_grid],
                                   all_data->vector.r[fine_grid]);
         all_data->output.residual_wtime[tid] += omp_get_wtime() - residual_start;
         restrict_start = omp_get_wtime();
         SMEM_Sync_Parfor_MatVec(all_data,
                                 all_data->matrix.R[fine_grid],
                                 all_data->vector.r[fine_grid],
                                 all_data->vector.f[coarse_grid]);
         all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
            all_data->vector.u[coarse_grid][i] = 0;
         }
      }
   }

   int coarsest_level = all_data->grid.num_levels-1;
   double smooth_start = omp_get_wtime();
   PARDISO(all_data->pardiso.info.pt,
           &(all_data->pardiso.info.maxfct),
           &(all_data->pardiso.info.mnum),
           &(all_data->pardiso.info.mtype),
           &(all_data->pardiso.info.phase),
           &(all_data->pardiso.csr.n),
           all_data->pardiso.csr.a,
           all_data->pardiso.csr.ia,
           all_data->pardiso.csr.ja,
           &(all_data->pardiso.info.idum),
           &(all_data->pardiso.info.nrhs),
           all_data->pardiso.info.iparm,
           &(all_data->pardiso.info.msglvl),
           all_data->vector.f[coarsest_level],
           all_data->vector.u[coarsest_level],
           &(all_data->pardiso.info.error));
   for (int t = 0; t < all_data->input.num_threads; t++){
      all_data->output.smooth_wtime[t] += omp_get_wtime() - smooth_start;
   }
   all_data->grid.local_num_correct[coarsest_level]++;

   #pragma omp parallel
   {
      int fine_grid, coarse_grid;
      int tid = omp_get_thread_num();

      double residual_start;
      double smooth_start;
      double restrict_start;
      double prolong_start;

      for (int level = all_data->grid.num_levels-2; level > -1; level--){
	 all_data->grid.zero_flags[level] = 0;
         fine_grid = level;
         coarse_grid = level + 1;
         prolong_start = omp_get_wtime();
         SMEM_Sync_Parfor_MatVec(all_data,
                                 all_data->matrix.P[fine_grid],
                                 all_data->vector.u[coarse_grid],
                                 all_data->vector.e[fine_grid]);
         all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
            all_data->vector.u[fine_grid][i] += all_data->vector.e[fine_grid][i];
         }
         smooth_start = omp_get_wtime();
         SMEM_Smooth(all_data,
                     all_data->matrix.A[fine_grid],
                     all_data->vector.f[fine_grid],
                     all_data->vector.u[fine_grid],
                     all_data->vector.u_prev[fine_grid],
                     all_data->vector.y[fine_grid],
                     all_data->input.num_post_smooth_sweeps,
                     fine_grid,
                     0, 0);
         all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
         if (tid == 0){
            all_data->grid.local_num_correct[level]++;
         }
      }
   }
}

void SMEM_Sync_Parfor_AFACx_Vcycle(AllData *all_data)
{
   #pragma omp parallel
   {
      int fine_grid, coarse_grid;
      int tid = omp_get_thread_num();

      double residual_start;
      double smooth_start;
      double restrict_start;
      double prolong_start;

      restrict_start = omp_get_wtime();
      for (int level = 0; level < all_data->grid.num_levels-1; level++){
         fine_grid = level;
         coarse_grid = level + 1;
         SMEM_Sync_Parfor_MatVec(all_data,
                                 all_data->matrix.R[fine_grid],
                                 all_data->vector.r[fine_grid],
                                 all_data->vector.r[coarse_grid]);
      }
      all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
   }

   int coarsest_level = all_data->grid.num_levels-1;
   double smooth_start = omp_get_wtime();
   PARDISO(all_data->pardiso.info.pt,
           &(all_data->pardiso.info.maxfct),
           &(all_data->pardiso.info.mnum),
           &(all_data->pardiso.info.mtype),
           &(all_data->pardiso.info.phase),
           &(all_data->pardiso.csr.n),
           all_data->pardiso.csr.a,
           all_data->pardiso.csr.ia,
           all_data->pardiso.csr.ja,
           &(all_data->pardiso.info.idum),
           &(all_data->pardiso.info.nrhs),
           all_data->pardiso.info.iparm,
           &(all_data->pardiso.info.msglvl),
           all_data->vector.r[coarsest_level],
           all_data->vector.u_fine[coarsest_level],
           &(all_data->pardiso.info.error));
   for (int t = 0; t < all_data->input.num_threads; t++){
      all_data->output.smooth_wtime[t] += omp_get_wtime() - smooth_start;
   }
   all_data->grid.local_num_correct[coarsest_level]++;

   #pragma omp parallel
   {
      int fine_grid, coarse_grid, thread_level;
      int tid = omp_get_thread_num();

      double residual_start;
      double smooth_start;
      double restrict_start;
      double prolong_start;

      for (int level = all_data->grid.num_levels-1; level > -1; level--){
	 all_data->grid.zero_flags[level] = 1;
         if (level != all_data->grid.num_levels-1){
            fine_grid = level;
            coarse_grid = level + 1;

            #pragma omp for
            for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
               all_data->vector.u_fine[fine_grid][i] = 0;
            }
            #pragma omp for
            for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
               all_data->vector.u_coarse[coarse_grid][i] = 0;
            }

            smooth_start = omp_get_wtime();
            SMEM_Smooth(all_data,
                        all_data->matrix.A[coarse_grid],
                        all_data->vector.r[coarse_grid],
                        all_data->vector.u_coarse[coarse_grid],
                        all_data->vector.u_coarse_prev[coarse_grid],
                        all_data->vector.y[coarse_grid],
                        all_data->input.num_coarse_smooth_sweeps,
                        coarse_grid,
                        0, 0);
            SMEM_Sync_Parfor_MatVec(all_data,
                                    all_data->matrix.P[fine_grid],
                                    all_data->vector.u_coarse[coarse_grid],
                                    all_data->vector.e[fine_grid]);
            SMEM_Sync_Parfor_Residual(all_data,
                                      all_data->matrix.A[fine_grid],
                                      all_data->vector.r[fine_grid],
                                      all_data->vector.e[fine_grid],
                                      all_data->vector.y[fine_grid],
                                      all_data->vector.r_fine[fine_grid]);
            SMEM_Smooth(all_data,
                        all_data->matrix.A[fine_grid],
                        all_data->vector.r_fine[fine_grid],
                        all_data->vector.u_fine[fine_grid],
                        all_data->vector.u_fine_prev[fine_grid],
                        all_data->vector.y[fine_grid],
                        all_data->input.num_fine_smooth_sweeps,
                        fine_grid,
                        0, 0);
            all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
            if (tid == 0){
               all_data->grid.local_num_correct[level]++;
            }
         }

         thread_level = level;
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[thread_level]; i++){
            all_data->vector.e[thread_level][i] = all_data->vector.u_fine[thread_level][i];
         }
         if (level > 0){
            for (int inner_level = level; inner_level > 0; inner_level--){
               fine_grid = inner_level - 1;
               coarse_grid = inner_level;
               prolong_start = omp_get_wtime();
               SMEM_Sync_Parfor_MatVec(all_data,
                                       all_data->matrix.P[fine_grid],
                                       all_data->vector.e[coarse_grid],
                                       all_data->vector.e[fine_grid]);
               all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
            }
         }
         fine_grid = 0;
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
            all_data->vector.u[fine_grid][i] += all_data->vector.e[fine_grid][i];
         }
      }
   }
}

void SMEM_Sync_Add_Vcycle(AllData *all_data)
{
   omp_init_lock(&(all_data->thread.lock));
   #pragma omp parallel
   {
      int tid = omp_get_thread_num();
      int fine_grid, coarse_grid;
      int thread_level;
      int ns, ne;

      double residual_start;
      double smooth_start;
      double restrict_start;
      double prolong_start;

      if (all_data->input.res_compute_type == GLOBAL){
	 all_data->grid.global_smooth_flags[tid] = 1;
         fine_grid = 0;
         thread_level = all_data->thread.thread_levels[tid][0];
         ns = all_data->thread.A_ns_global[tid];
         ne = all_data->thread.A_ne_global[tid];
         for (int i = ns; i < ne; i++){
            all_data->level_vector[thread_level].e[fine_grid][i] = 0;
         }
         smooth_start = omp_get_wtime();
	 if (all_data->input.solver == MULTADD){
            SMEM_Smooth(all_data,
                        all_data->matrix.A[fine_grid],
                        all_data->vector.r[fine_grid],
                        all_data->level_vector[thread_level].e[fine_grid],
                        all_data->level_vector[thread_level].u_prev[fine_grid],
                        all_data->level_vector[thread_level].y[fine_grid],
                        all_data->input.num_fine_smooth_sweeps,
                        thread_level,
                        ns, ne);
            all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
            for (int i = ns; i < ne; i++){
               #pragma omp atomic
               all_data->vector.u[fine_grid][i] += all_data->level_vector[thread_level].e[fine_grid][i];
            }
	    all_data->grid.global_smooth_flags[tid] = 0;
	    #pragma omp barrier
	 }
      }

      for (int q = 0; q < all_data->thread.thread_levels[tid].size(); q++){
         thread_level = all_data->thread.thread_levels[tid][q];
	 if (tid == all_data->thread.barrier_root[thread_level]){
            all_data->grid.zero_flags[thread_level] = 1;
	 }

         int coarsest_level;
         if (all_data->input.solver == MULTADD){
            coarsest_level = thread_level;
         }
         else{
            coarsest_level = thread_level+1;
         }

         fine_grid = 0;
         ns = all_data->thread.A_ns[fine_grid][tid];
         ne = all_data->thread.A_ne[fine_grid][tid];
         for (int i = ns; i < ne; i++){
            all_data->level_vector[thread_level].r[fine_grid][i] = all_data->vector.r[fine_grid][i];
         }
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level); 
         for (int level = 0; level < coarsest_level; level++){
            if (level < all_data->grid.num_levels-1){
               fine_grid = level;
               coarse_grid = level + 1;
               ns = all_data->thread.R_ns[fine_grid][tid];
               ne = all_data->thread.R_ne[fine_grid][tid];
               restrict_start = omp_get_wtime();
               SMEM_MatVec(all_data,
                           all_data->matrix.R[fine_grid],
                           all_data->level_vector[thread_level].r[fine_grid],
                           all_data->level_vector[thread_level].r[coarse_grid],
                           ns, ne);
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
            }
         }
         fine_grid = thread_level;
         coarse_grid = thread_level + 1;
         smooth_start = omp_get_wtime();
         if (thread_level == all_data->grid.num_levels-1){
            if (tid == all_data->thread.level_threads[thread_level][0]){
               PARDISO(all_data->pardiso.info.pt,
                       &(all_data->pardiso.info.maxfct),
                       &(all_data->pardiso.info.mnum),
                       &(all_data->pardiso.info.mtype),
                       &(all_data->pardiso.info.phase),
                       &(all_data->pardiso.csr.n),
                       all_data->pardiso.csr.a,
                       all_data->pardiso.csr.ia,
                       all_data->pardiso.csr.ja,
                       &(all_data->pardiso.info.idum),
                       &(all_data->pardiso.info.nrhs),
                       all_data->pardiso.info.iparm,
                       &(all_data->pardiso.info.msglvl),
                       all_data->level_vector[thread_level].r[thread_level],
                       all_data->level_vector[thread_level].e[thread_level],
                       &(all_data->pardiso.info.error));
            }
            SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
         }
         else {
            if (all_data->input.solver == MULTADD){
               ns = all_data->thread.A_ns[thread_level][tid];
               ne = all_data->thread.A_ne[thread_level][tid];
               for (int i = ns; i < ne; i++){
                  all_data->level_vector[thread_level].e[thread_level][i] = 0;
               }
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               SMEM_Smooth(all_data,
                              all_data->matrix.A[thread_level],
                              all_data->level_vector[thread_level].r[thread_level],
                              all_data->level_vector[thread_level].e[thread_level],
                              all_data->level_vector[thread_level].u_prev[thread_level],
                              all_data->level_vector[thread_level].y[thread_level],
                              all_data->input.num_fine_smooth_sweeps,
                              thread_level,
                              ns, ne);
            }
            else{
               ns = all_data->thread.A_ns[fine_grid][tid];
               ne = all_data->thread.A_ne[fine_grid][tid];
               for (int i = ns; i < ne; i++){
                  all_data->level_vector[thread_level].u_fine[fine_grid][i] = 0;
               }
               ns = all_data->thread.A_ns[coarse_grid][tid];
               ne = all_data->thread.A_ne[coarse_grid][tid];
               for (int i = ns; i < ne; i++){
                  all_data->level_vector[thread_level].u_coarse[coarse_grid][i] = 0;
               }
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               SMEM_Smooth(all_data,
                           all_data->matrix.A[coarse_grid],
                           all_data->level_vector[thread_level].r[coarse_grid],
                           all_data->level_vector[thread_level].u_coarse[coarse_grid],
                           all_data->level_vector[thread_level].u_coarse_prev[coarse_grid],
                           all_data->level_vector[thread_level].y[coarse_grid],
                           all_data->input.num_coarse_smooth_sweeps,
                           thread_level,
                           ns, ne);
               ns = all_data->thread.A_ns[fine_grid][tid];
               ne = all_data->thread.A_ne[fine_grid][tid];
               SMEM_MatVec(all_data,
                           all_data->matrix.P[fine_grid],
                           all_data->level_vector[thread_level].u_coarse[coarse_grid],
                           all_data->level_vector[thread_level].e[fine_grid],
                           ns, ne);
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               SMEM_Residual(all_data,
                             all_data->matrix.A[fine_grid],
                             all_data->level_vector[thread_level].r[fine_grid],
                             all_data->level_vector[thread_level].e[fine_grid],
                             all_data->level_vector[thread_level].y[fine_grid],
                             all_data->level_vector[thread_level].r_fine[fine_grid],
                             ns, ne);
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               SMEM_Smooth(all_data,
                           all_data->matrix.A[fine_grid],
                           all_data->level_vector[thread_level].r_fine[fine_grid],
                           all_data->level_vector[thread_level].u_fine[fine_grid],
                           all_data->level_vector[thread_level].u_fine_prev[fine_grid],
                           all_data->level_vector[thread_level].y[fine_grid],
                           all_data->input.num_fine_smooth_sweeps,
                           thread_level,
                           ns, ne);
               ns = all_data->thread.A_ns[thread_level][tid];
               ne = all_data->thread.A_ne[thread_level][tid];
               for (int i = ns; i < ne; i++){
                  all_data->level_vector[thread_level].e[thread_level][i] =
                     all_data->level_vector[thread_level].u_fine[thread_level][i];
               }
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
            }
         }
         all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;

         if (thread_level > 0){
            for (int level = thread_level-1; level > -1; level--){
               fine_grid = level;
               coarse_grid = level + 1;
               ns = all_data->thread.P_ns[fine_grid][tid];
               ne = all_data->thread.P_ne[fine_grid][tid];
               prolong_start = omp_get_wtime();
               SMEM_MatVec(all_data,
                           all_data->matrix.P[fine_grid],
                           all_data->level_vector[thread_level].e[coarse_grid],
                           all_data->level_vector[thread_level].e[fine_grid],
                           ns, ne);
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
            }
         }
       
         fine_grid = 0;
         ns = all_data->thread.A_ns[fine_grid][tid];
         ne = all_data->thread.A_ne[fine_grid][tid];
        // if (all_data->input.async_type == SEMI_ASYNC){
        //    if (tid == all_data->thread.barrier_root[thread_level]){
        //       omp_set_lock(&(all_data->thread.lock));
        //       all_data->grid.global_num_correct++;
        //    }
        //    SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
        //    for (int i = ns; i < ne; i++){
        //       all_data->vector.u[fine_grid][i] += all_data->level_vector[thread_level].e[fine_grid][i];
        //    }
        //    if (tid == all_data->thread.barrier_root[thread_level]){
        //       omp_unset_lock(&(all_data->thread.lock));
        //    }
        // }
        // else {
            for (int i = ns; i < ne; i++){
               #pragma omp atomic
               all_data->vector.u[fine_grid][i] += all_data->level_vector[thread_level].e[fine_grid][i];
            }
        // }
         if (tid == all_data->thread.barrier_root[thread_level]){
            all_data->grid.local_num_correct[thread_level]++;
         }
      }
   }
   omp_destroy_lock(&(all_data->thread.lock));
}
