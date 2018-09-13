#include "Main.hpp"
#include "SMEM_MatVec.hpp"
#include "SMEM_Smooth.hpp"
#include "SMEM_Solve.hpp"
#include "Misc.hpp"

void SMEM_Async_Add_AMG(AllData *all_data)
{
   all_data->grid.res_comp_count = 0;
   omp_init_lock(&(all_data->thread.lock));
   #pragma omp parallel
   {
      int tid = omp_get_thread_num();
      int fine_grid, coarse_grid;
      int thread_level;
      int tid_converge = 0;
      int ns, ne;

      double residual_start;
      double smooth_start;
      double restrict_start;
      double prolong_start;

      while(1){
         for (int q = 0; q < all_data->thread.thread_levels[tid].size(); q++){
            thread_level = all_data->thread.thread_levels[tid][q];
            fine_grid = 0;
            ns = all_data->thread.A_ns[fine_grid][tid];
            ne = all_data->thread.A_ne[fine_grid][tid];
            if (tid == 0){
               all_data->output.r_norm2 = 0;
            }
            residual_start = omp_get_wtime();
            SMEM_Residual(all_data,
                          all_data->matrix.A[fine_grid],
                          all_data->vector.f[fine_grid],
                          all_data->level_vector[thread_level].u[fine_grid],
                          all_data->level_vector[thread_level].y[fine_grid],
                          all_data->level_vector[thread_level].r[fine_grid],
                          ns, ne);
            SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
            all_data->output.residual_wtime[tid] += omp_get_wtime() - residual_start;

            if (thread_level == 0 &&
                all_data->input.check_resnorm_flag == 1){
               Par_Norm2(all_data,
                         all_data->level_vector[thread_level].r[fine_grid],
                         thread_level,
                         ns, ne);
            }
            if (tid == 0){
               all_data->thread.converge_flag = CheckConverge(all_data, thread_level);
            }

            int coarsest_level;
            if (all_data->input.solver == ASYNC_MULTADD){
               coarsest_level = thread_level;
            }
            else{
               coarsest_level = thread_level+1;
            }
            
            for (int level = 0; level < coarsest_level; level++){
               if (level < all_data->grid.num_levels-1){
                  fine_grid = level;
                  coarse_grid = level + 1;
                  if (all_data->input.solver == ASYNC_MULTADD){
                     ns = all_data->thread.A_ns[fine_grid][tid];
                     ne = all_data->thread.A_ne[fine_grid][tid];
                     smooth_start = omp_get_wtime();
                     SMEM_JacobiIterMat_MatVec(all_data,
                                               all_data->matrix.A[fine_grid],
                                               all_data->level_vector[thread_level].y[fine_grid],
                                               all_data->level_vector[thread_level].r[fine_grid],
                                               ns, ne,
                                               thread_level);
                     SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
                     all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
                  }
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
               if (all_data->input.solver == ASYNC_MULTADD){
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
               smooth_start = omp_get_wtime();
               if (all_data->input.solver == ASYNC_MULTADD){
                  ns = all_data->thread.A_ns[fine_grid][tid];
                  ne = all_data->thread.A_ne[fine_grid][tid];
                  SMEM_JacobiIterMat_MatVec(all_data,
                                            all_data->matrix.A[fine_grid],
                                            all_data->level_vector[thread_level].y[fine_grid],
                                            all_data->level_vector[thread_level].e[fine_grid],
                                            ns, ne,
                                            thread_level);
                  SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               }
               all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
            }
          
            fine_grid = 0;
            ns = all_data->thread.A_ns[fine_grid][tid];
            ne = all_data->thread.A_ne[fine_grid][tid];
            if (all_data->input.async_type == SEMI_ASYNC){
               if (tid == all_data->thread.barrier_root[thread_level]){
                  omp_set_lock(&(all_data->thread.lock));
               }
               SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
               for (int i = ns; i < ne; i++){
                  all_data->vector.u[fine_grid][i] += all_data->level_vector[thread_level].e[fine_grid][i];
                  all_data->level_vector[thread_level].u[fine_grid][i] = all_data->vector.u[fine_grid][i];
               }
               if (tid == all_data->thread.barrier_root[thread_level]){
                  omp_unset_lock(&(all_data->thread.lock));
               }
            }
            else {
               for (int i = ns; i < ne; i++){
                  #pragma omp atomic
                  all_data->vector.u[fine_grid][i] += all_data->level_vector[thread_level].e[fine_grid][i];
                  all_data->level_vector[thread_level].u[fine_grid][i] = all_data->vector.u[fine_grid][i];
               }
            }

            if (tid == all_data->thread.barrier_root[thread_level]){
               all_data->grid.num_correct[thread_level]++;
            }
            if (tid == 0){
               all_data->thread.converge_flag = CheckConverge(all_data, thread_level);
            }
            if (SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level) == 1){
               tid_converge = 1;
            }
            if (all_data->input.converge_test_type == ONE_LEVEL){
               if (all_data->grid.num_correct[thread_level] == all_data->input.num_cycles){
                  tid_converge = 1;
               }
            }
         }
         if (tid_converge == 1){
            break;
         }
      }
   }
   omp_destroy_lock(&(all_data->thread.lock));
  // for (int level = 0; level < all_data->grid.num_levels; level++){
  //    printf("level %d: %d\n", level, all_data->grid.num_correct[level]);
  // }
}
