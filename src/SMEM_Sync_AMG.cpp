#include "Main.hpp"
#include "SMEM_MatVec.hpp"
#include "SMEM_Smooth.hpp"
#include "Misc.hpp"
#include "SMEM_Solve.hpp"


void SMEM_Sync_Parfor_Restrict(AllData *all_data,
                               hypre_CSRMatrix *R,
                               HYPRE_Real *v_fine,
                               HYPRE_Real *v_coarse,
                               int fine_grid, int coarse_grid);

void SMEM_Sync_Parfor_Vcycle(AllData *all_data)
{
   int fine_grid, coarse_grid;
   int tid = omp_get_thread_num();

   double residual_start;
   double smooth_start;
   double restrict_start;
   double prolong_start;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;

   for (int level = 0; level < all_data->grid.num_levels-1; level++){
      fine_grid = level;
      coarse_grid = level + 1;

      all_data->grid.zero_flags[level] = 1;
      if (level == 0 && all_data->input.precond_flag == 0){
         all_data->grid.zero_flags[level] = 0;
      }
  
      HYPRE_Real *f_fine;
      if (fine_grid == 0 && all_data->input.precond_flag == 1){
         f_fine = all_data->vector.r[fine_grid];
      }
      else {
         f_fine = all_data->vector.f[fine_grid];
      }

      smooth_start = omp_get_wtime();
      SMEM_Smooth(all_data,
                  all_data->matrix.A[fine_grid],
                  f_fine,
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
                                f_fine,
                                all_data->vector.u[fine_grid],
                                all_data->vector.y[fine_grid],
                                all_data->vector.r_fine[fine_grid]);
      all_data->output.residual_wtime[tid] += omp_get_wtime() - residual_start;
      restrict_start = omp_get_wtime();
      SMEM_Sync_Parfor_Restrict(all_data,
                                all_data->matrix.R[fine_grid],
                                all_data->vector.r_fine[fine_grid],
                                all_data->vector.f[coarse_grid],
                                fine_grid, coarse_grid);
      all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
     // #pragma omp for
     // for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
     //    all_data->vector.u[coarse_grid][i] = 0;
     // }
   }
   int coarsest_level = all_data->grid.num_levels-1;
   if (tid == 0){
      smooth_start = omp_get_wtime();
      for (int i = 0; i < all_data->grid.n[coarsest_level]; i++){
         hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[coarsest_level]))[i] = all_data->vector.f[coarsest_level][i];
      }
      hypre_GaussElimSolve(amg_data, coarsest_level, 9);
      for (int i = 0; i < all_data->grid.n[coarsest_level]; i++){
         all_data->vector.u[coarsest_level][i] = hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataUArray(amg_data)[coarsest_level]))[i];
      }
      all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
   }
   #pragma omp barrier

   for (int level = all_data->grid.num_levels-2; level > -1; level--){
      all_data->grid.zero_flags[level] = 0;
      fine_grid = level;
      coarse_grid = level + 1;
     // prolong_start = omp_get_wtime();
     // SMEM_Sync_Parfor_MatVec(all_data,
     //                         all_data->matrix.P[fine_grid],
     //                         all_data->vector.u[coarse_grid],
     //                         all_data->vector.e[fine_grid]);
     // all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
     // #pragma omp for
     // for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
     //    all_data->vector.u[fine_grid][i] += all_data->vector.e[fine_grid][i];
     // }
      prolong_start = omp_get_wtime();
      SMEM_Sync_Parfor_SpGEMV(all_data,
                              all_data->matrix.P[fine_grid],
                              all_data->vector.u[coarse_grid],
                              all_data->vector.u[fine_grid],
                              1.0, 1.0,
                              all_data->vector.u[fine_grid]);
      all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
      smooth_start = omp_get_wtime();
      HYPRE_Real *f_fine;
      if (fine_grid == 0 && all_data->input.precond_flag == 1){
         f_fine = all_data->vector.r[fine_grid];
      }
      else {
         f_fine = all_data->vector.f[fine_grid];
      }
      SMEM_Smooth(all_data,
                  all_data->matrix.A[fine_grid],
                  f_fine,
                  all_data->vector.u[fine_grid],
                  all_data->vector.u_prev[fine_grid],
                  all_data->vector.y[fine_grid],
                  all_data->input.num_post_smooth_sweeps,
                  fine_grid,
                  0, 0);
      all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
   }
}

void SMEM_Sync_Parfor_BPXcycle(AllData *all_data)
{
   int fine_grid, coarse_grid;
   int tid = omp_get_thread_num();

   double residual_start;
   double smooth_start;
   double restrict_start;
   double prolong_start;
   int thread_level;
   int *disp = all_data->grid.disp;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;

  // if (all_data->input.precond_flag == 1){
  //    if (all_data->input.solver == PAR_BPX){
  //       #pragma omp for
  //       for (int i = 0; i < all_data->grid.n[0]; i++){
  //          all_data->vector.u[0][i] = 0;
  //          all_data->vector.xx[i] = 0;
  //          //all_data->vector.rr[i] = all_data->vector.r[0][i];
  //       }
  //    }
  //    else {
  //       #pragma omp for
  //       for (int i = 0; i < all_data->grid.n[0]; i++){
  //          all_data->vector.u[0][i] = 0;
  //       }
  //    }
  // }

   restrict_start = omp_get_wtime();
   for (int level = 0; level < all_data->grid.num_levels-1; level++){
      fine_grid = level;
      coarse_grid = level + 1;
      HYPRE_Real *r_fine, *r_coarse;
      if (all_data->input.solver == PAR_BPX){
         if (level == 0){
            r_fine = &(all_data->vector.rr[disp[fine_grid]]);
         }
         else {
            r_fine = &(all_data->vector.rr[disp[fine_grid]]);
         }
         r_coarse = &(all_data->vector.rr[disp[coarse_grid]]);
      }
      else {
         r_fine = all_data->vector.r[fine_grid];
         r_coarse = all_data->vector.r[coarse_grid];
      }
      SMEM_Sync_Parfor_Restrict(all_data,
                                all_data->matrix.R[fine_grid],
                                r_fine,
                                r_coarse,
                                fine_grid, coarse_grid);
   }
   all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;

   int coarsest_level = all_data->grid.num_levels-1;
   smooth_start = omp_get_wtime();
   if (all_data->input.solver == PAR_BPX){
      #pragma omp for
      for (int i = 0; i < all_data->grid.N; i++){
         all_data->vector.xx[i] = all_data->vector.rr[i] / all_data->matrix.A_diag_ext[i];
      }
   }
   else {
      for (int level = 0; level < all_data->grid.num_levels; level++){
         all_data->grid.zero_flags[level] = 1;
        // if (level == all_data->grid.num_levels-1){
        //    if (tid == 0){
        //       hypre_GaussElimSolve(amg_data, coarsest_level, 9);
        //    }
        // }
        // else {
            SMEM_Smooth(all_data,
                        all_data->matrix.A[level],
                        all_data->vector.r[level],
                        all_data->vector.e[level],
                        all_data->vector.u_prev[level],
                        all_data->vector.y[level],
                        all_data->input.num_pre_smooth_sweeps,
                        level,
                        0, 0);
        // }
      }
   }
   all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;

   prolong_start = omp_get_wtime();
   for (int level = all_data->grid.num_levels-2; level > -1; level--){
      fine_grid = level;
      coarse_grid = level + 1;
      HYPRE_Real *e_fine, *e_coarse;
      if (all_data->input.solver == PAR_BPX){
         e_fine = &(all_data->vector.xx[disp[fine_grid]]);
         e_coarse = &(all_data->vector.xx[disp[coarse_grid]]);
      }
      else {
         e_fine = all_data->vector.e[fine_grid];
         e_coarse = all_data->vector.e[coarse_grid];
      }
     // SMEM_Sync_Parfor_MatVec(all_data,
     //                         all_data->matrix.P[fine_grid],
     //                         e_coarse,
     //                         all_data->vector.y[fine_grid]);
     // #pragma omp for
     // for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
     //    e_fine[i] += all_data->vector.y[fine_grid][i];
     // }
      SMEM_Sync_Parfor_SpGEMV(all_data,
                              all_data->matrix.P[fine_grid],
                              e_coarse,
                              e_fine,
                              1.0, 1.0,
                              e_fine);
   }
   all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;


   HYPRE_Real *e;
   if (all_data->input.solver == PAR_BPX){
      e = all_data->vector.xx;
   }
   else {
      e = all_data->vector.e[0];
   }
   if (all_data->input.precond_flag == 1){
      #pragma omp for
      for (int i = 0; i < all_data->grid.n[0]; i++){
         all_data->vector.u[0][i] = e[i];
      }
   }
   else {
      #pragma omp for
      for (int i = 0; i < all_data->grid.n[0]; i++){
         all_data->vector.u[0][i] += e[i];
      }
   }
}

void SMEM_Sync_Parfor_AFACx_Vcycle(AllData *all_data)
{
   int fine_grid, coarse_grid;
   int tid = omp_get_thread_num();

   double residual_start;
   double smooth_start;
   double restrict_start;
   double prolong_start;
   int thread_level;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;

   restrict_start = omp_get_wtime();
   for (int level = 0; level < all_data->grid.num_levels-1; level++){
      fine_grid = level;
      coarse_grid = level + 1;
      SMEM_Sync_Parfor_Restrict(all_data,
                                all_data->matrix.R[fine_grid],
                                all_data->vector.r[fine_grid],
                                all_data->vector.r[coarse_grid],
                                fine_grid, coarse_grid);
   }
   all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;

   int coarsest_level = all_data->grid.num_levels-1;
   smooth_start = omp_get_wtime();
   if (tid == 0){
      hypre_GaussElimSolve(amg_data, coarsest_level, 9);
   }
   #pragma omp barrier
   for (int t = 0; t < all_data->input.num_threads; t++){
      all_data->output.smooth_wtime[t] += omp_get_wtime() - smooth_start;
   }
   all_data->grid.local_num_correct[coarsest_level]++;

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

      #pragma omp for
      for (int i = 0; i < all_data->grid.n[level]; i++){
         all_data->vector.e[level][i] = all_data->vector.u_fine[level][i];
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

void SMEM_Sync_Add_Vcycle(AllData *all_data)
{
   int tid = omp_get_thread_num();
   int fine_grid, coarse_grid;
   int thread_level;
   int ns, ne;

   double residual_start;
   double smooth_start;
   double restrict_start;
   double prolong_start;
   
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;

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
           // #pragma omp atomic
            all_data->vector.u[fine_grid][i] += all_data->level_vector[thread_level].e[fine_grid][i];
         }
         all_data->grid.global_smooth_flags[tid] = 0;
         #pragma omp barrier
        // SMEM_Barrier(all_data, all_data->thread.global_barrier_flags);
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
            hypre_GaussElimSolve(amg_data, coarsest_level, 9);
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
      if (all_data->input.async_type == SEMI_ASYNC){
         if (tid == all_data->thread.barrier_root[thread_level]){
            omp_set_lock(&(all_data->thread.lock));
            all_data->grid.global_num_correct++;
         }
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
         for (int i = ns; i < ne; i++){
            all_data->vector.u[fine_grid][i] += all_data->level_vector[thread_level].e[fine_grid][i];
         }
         if (tid == all_data->thread.barrier_root[thread_level]){
            omp_unset_lock(&(all_data->thread.lock));
         }
      }
      else {
         for (int i = ns; i < ne; i++){
           // #pragma omp atomic
            all_data->vector.u[fine_grid][i] += all_data->level_vector[thread_level].e[fine_grid][i];
         }
      }
      if (tid == all_data->thread.barrier_root[thread_level]){
         all_data->grid.local_num_correct[thread_level]++;
      }
   }
}


void SMEM_Sync_Parfor_Restrict(AllData *all_data,
                               hypre_CSRMatrix *R,
                               HYPRE_Real *v_fine,
                               HYPRE_Real *v_coarse,
                               int fine_grid, int coarse_grid)
{
   if (all_data->input.construct_R_flag == 1){
      SMEM_Sync_Parfor_MatVec(all_data, R, v_fine, v_coarse);
   }
   else {
      SMEM_Sync_Parfor_MatVecT(all_data, R, v_fine, v_coarse, all_data->vector.y_extend[fine_grid]);
   }
}
