#include "Main.hpp"
#include "SEQ_Smooth.hpp"
#include "SEQ_MatVec.hpp"
#include "Misc.hpp"
#include "SMEM_Solve.hpp"

using namespace std;

void SEQ_Vcycle(AllData *all_data)
{
   int fine_grid, coarse_grid, this_grid;
   int tid = 0;

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
      SEQ_Residual(all_data,
                   all_data->matrix.A[fine_grid],
                   all_data->vector.f[fine_grid],
                   all_data->vector.u[fine_grid],
                   all_data->vector.y[fine_grid],
                   all_data->vector.r[fine_grid]);
      all_data->output.residual_wtime[tid] += omp_get_wtime() - residual_start;
      restrict_start = omp_get_wtime();
      SEQ_MatVec(all_data,
                 all_data->matrix.R[fine_grid],
                 all_data->vector.r[fine_grid],
                 all_data->vector.f[coarse_grid]);
      all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
      for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
         all_data->vector.u[coarse_grid][i] = 0;
      }
   }

   this_grid = all_data->grid.num_levels-1;
   smooth_start = omp_get_wtime();
  // PARDISO(all_data->pardiso.info.pt,
  //         &(all_data->pardiso.info.maxfct),
  //         &(all_data->pardiso.info.mnum),
  //         &(all_data->pardiso.info.mtype),
  //         &(all_data->pardiso.info.phase),
  //         &(all_data->pardiso.csr.n),
  //         all_data->pardiso.csr.a,
  //         all_data->pardiso.csr.ia,
  //         all_data->pardiso.csr.ja,
  //         &(all_data->pardiso.info.idum),
  //         &(all_data->pardiso.info.nrhs),
  //         all_data->pardiso.info.iparm,
  //         &(all_data->pardiso.info.msglvl),
  //         all_data->vector.f[this_grid],
  //         all_data->vector.u[this_grid],
  //         &(all_data->pardiso.info.error));
   all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
   all_data->grid.local_num_correct[this_grid]++;


   for (int level = all_data->grid.num_levels-2; level > -1; level--){
      all_data->grid.zero_flags[level] = 0;
      fine_grid = level;
      coarse_grid = level + 1;
      prolong_start = omp_get_wtime();
      SEQ_MatVec(all_data,
                 all_data->matrix.P[fine_grid],
                 all_data->vector.u[coarse_grid],
                 all_data->vector.e[fine_grid]);
      all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
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
      all_data->grid.local_num_correct[level]++;
   }
}

void SEQ_Add_Vcycle(AllData *all_data)
{
   int fine_grid, coarse_grid, this_grid;
   int tid = 0;

   double residual_start;
   double smooth_start;
   double restrict_start;
   double prolong_start;

   for (int level = 0; level < all_data->grid.num_levels-1; level++){
      all_data->grid.zero_flags[level] = 1;
      fine_grid = level;
      coarse_grid = level + 1;
      restrict_start = omp_get_wtime();
      SEQ_MatVec(all_data,
                 all_data->matrix.R[fine_grid],
                 all_data->vector.r[fine_grid],
                 all_data->vector.r[coarse_grid]);
      all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
   }

   smooth_start = omp_get_wtime();
   for (int level = 0; level < all_data->grid.num_levels; level++){
      if (level == all_data->grid.num_levels-1){
         this_grid = all_data->grid.num_levels-1;
        // PARDISO(all_data->pardiso.info.pt,
        //         &(all_data->pardiso.info.maxfct),
        //         &(all_data->pardiso.info.mnum),
        //         &(all_data->pardiso.info.mtype),
        //         &(all_data->pardiso.info.phase),
        //         &(all_data->pardiso.csr.n),
        //         all_data->pardiso.csr.a,
        //         all_data->pardiso.csr.ia,
        //         all_data->pardiso.csr.ja,
        //         &(all_data->pardiso.info.idum),
        //         &(all_data->pardiso.info.nrhs),
        //         all_data->pardiso.info.iparm,
        //         &(all_data->pardiso.info.msglvl),
        //         all_data->vector.r[this_grid],
        //         all_data->vector.u_fine[this_grid],
        //         &(all_data->pardiso.info.error));
      }
      else {
         fine_grid = level;
         coarse_grid = level + 1;

         if (all_data->input.solver == MULTADD){
            for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
               all_data->vector.u_fine[fine_grid][i] = 0;
            }
            SMEM_Smooth(all_data,
                        all_data->matrix.A[fine_grid],
                        all_data->vector.r[fine_grid],
                        all_data->vector.u_fine[fine_grid],
                        all_data->vector.u_fine_prev[fine_grid],
                        all_data->vector.y[fine_grid],
                        all_data->input.num_fine_smooth_sweeps,
                        fine_grid,
                        0, 0); 
         }
         else{
            for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
               all_data->vector.u_fine[fine_grid][i] = 0;
            }
            for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
               all_data->vector.u_coarse[coarse_grid][i] = 0;
            }
            SMEM_Smooth(all_data,
                        all_data->matrix.A[coarse_grid],
                        all_data->vector.r[coarse_grid],
                        all_data->vector.u_coarse[coarse_grid],
                        all_data->vector.u_coarse_prev[coarse_grid],
                        all_data->vector.y[coarse_grid],
                        all_data->input.num_coarse_smooth_sweeps,
                        coarse_grid,
                        0, 0);
            SEQ_MatVec(all_data,
                       all_data->matrix.P[fine_grid],
                       all_data->vector.u_coarse[coarse_grid],
                       all_data->vector.e[fine_grid]);
            SEQ_Residual(all_data,
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
         }
      }
      all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;
      all_data->grid.local_num_correct[level]++;

      this_grid = level;
      for (int i = 0; i < all_data->grid.n[this_grid]; i++){
         all_data->vector.e[this_grid][i] = all_data->vector.u_fine[this_grid][i];
      }
      if (level > 0){
         for (int inner_level = level; inner_level > 0; inner_level--){
            fine_grid = inner_level - 1;
            coarse_grid = inner_level;
            prolong_start = omp_get_wtime();
            SEQ_MatVec(all_data,
                       all_data->matrix.P[fine_grid],
                       all_data->vector.e[coarse_grid],
                       all_data->vector.e[fine_grid]);
            all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
         }
      }
      fine_grid = 0;
      for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
         all_data->vector.u[fine_grid][i] += all_data->vector.e[fine_grid][i];
      }
   }
}

void SEQ_Add_Vcycle_Sim(AllData *all_data)
{
   srand(time(NULL));
   int fine_grid, coarse_grid, this_grid;
   int tid = 0;

   double residual_start;
   double smooth_start;
   double restrict_start;
   double prolong_start; 

   vector<double> read_hist(all_data->grid.n[0],0);
   vector<vector<double>> e_write(all_data->grid.num_levels, vector<double>(all_data->grid.n[0]));
   double **read = (double **)calloc(all_data->grid.num_levels, sizeof(double *));
   vector<int> grid_wait_list(all_data->grid.num_levels);
   vector<int> grid_time_count(all_data->grid.num_levels);
   vector<int> correct_flags(all_data->grid.num_levels);

   srand(0);
   for (int level = 0; level < all_data->grid.num_levels; level++){
      read[level] = (double *)calloc(all_data->grid.n[fine_grid], sizeof(double));
      fine_grid = 0;
      grid_time_count[level] = 0;
      grid_wait_list[level] = (int)round(RandDouble(0.0, all_data->input.sim_grid_wait));
   }

   vector<int> level_perm(all_data->grid.num_levels);
   for (int level = 0; level < all_data->grid.num_levels; level++){
      level_perm[level] = level;
   }

   int num_cycles;
   if (all_data->input.converge_test_type == LOCAL){
      num_cycles = 1;
   }
   else {
      num_cycles = all_data->input.num_cycles;
   }

   srand(time(NULL));
   for (int k = 0; k < num_cycles; k++){
      while(1){
         for (int level = 0; level < all_data->grid.num_levels; level++){
	    all_data->grid.zero_flags[level] = 1;
            correct_flags[level] = 0;
            if (all_data->input.converge_test_type == LOCAL){
               if (grid_time_count[level] == grid_wait_list[level] &&
                   all_data->grid.local_num_correct[level] < all_data->input.num_cycles){
                  correct_flags[level] = 1;
               } 
            }
            else{
               if (grid_time_count[level] == grid_wait_list[level]){
                  correct_flags[level] = 1;
               }
            }
            if (correct_flags[level] == 1){
               if (all_data->input.async_type == FULL_ASYNC){ 
                  fine_grid = 0;
                  for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                     int rand_low = //max(0, all_data->grid.global_num_correct - all_data->input.sim_read_delay);
                        max(all_data->grid.last_read_correct[level],
                            all_data->grid.global_num_correct - all_data->input.sim_read_delay);
                     int rand_high = all_data->grid.global_num_correct;
                     int col = (int)round(RandDouble((double)rand_low, (double)rand_high)); 
                     col *= all_data->grid.n[0];
                     read[level][i] = read_hist[col+i];
                  }
               }

               fine_grid = 0;
               residual_start = omp_get_wtime();
               SEQ_Residual(all_data,
                            all_data->matrix.A[fine_grid],
                            all_data->vector.f[fine_grid],
                            read[level],
                            all_data->vector.y[fine_grid],
                            all_data->vector.r[fine_grid]);
               all_data->output.residual_wtime[tid] += omp_get_wtime() - residual_start; 
               int coarsest_level;
               if (all_data->input.solver == MULTADD ||
                   all_data->input.solver == ASYNC_MULTADD){
                  coarsest_level = level;
               }
               else{
                  coarsest_level = level+1;
               }

               for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
                  fine_grid = inner_level;
                  coarse_grid = inner_level + 1;
                  if (inner_level < all_data->grid.num_levels-1){
                     restrict_start = omp_get_wtime();
                     SEQ_MatVec(all_data,
                                all_data->matrix.R[fine_grid],
                                all_data->vector.r[fine_grid],
                                all_data->vector.r[coarse_grid]);
                     all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
                  }
               }
               smooth_start = omp_get_wtime();
               if (level == all_data->grid.num_levels-1){
                  this_grid = all_data->grid.num_levels-1;
                 // PARDISO(all_data->pardiso.info.pt,
                 //         &(all_data->pardiso.info.maxfct),
                 //         &(all_data->pardiso.info.mnum),
                 //         &(all_data->pardiso.info.mtype),
                 //         &(all_data->pardiso.info.phase),
                 //         &(all_data->pardiso.csr.n),
                 //         all_data->pardiso.csr.a,
                 //         all_data->pardiso.csr.ia,
                 //         all_data->pardiso.csr.ja,
                 //         &(all_data->pardiso.info.idum),
                 //         &(all_data->pardiso.info.nrhs),
                 //         all_data->pardiso.info.iparm,
                 //         &(all_data->pardiso.info.msglvl),
                 //         all_data->vector.r[this_grid],
                 //         all_data->vector.u_fine[this_grid],
                 //         &(all_data->pardiso.info.error));
               }
               else {
                  fine_grid = level;
                  coarse_grid = level + 1;

                  if (all_data->input.solver == MULTADD ||
                      all_data->input.solver == ASYNC_MULTADD){
                     for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                        all_data->vector.u_fine[fine_grid][i] = 0;
                     }
                     SMEM_Smooth(all_data,
                                 all_data->matrix.A[fine_grid],
                                 all_data->vector.r[fine_grid],
                                 all_data->vector.u_fine[fine_grid],
                                 all_data->vector.u_fine_prev[fine_grid],
                                 all_data->vector.y[fine_grid],
                                 all_data->input.num_fine_smooth_sweeps,
                                 fine_grid,
                                 0, 0); 
                  }
                  else{
                     for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                        all_data->vector.u_fine[fine_grid][i] = 0;
                     }
                     for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
                        all_data->vector.u_coarse[coarse_grid][i] = 0;
                     }
                     SMEM_Smooth(all_data,
                                 all_data->matrix.A[coarse_grid],
                                 all_data->vector.r[coarse_grid],
                                 all_data->vector.u_coarse[coarse_grid],
                                 all_data->vector.u_coarse_prev[coarse_grid],
                                 all_data->vector.y[coarse_grid],
                                 all_data->input.num_coarse_smooth_sweeps,
                                 coarse_grid,
                                 0, 0);
                     SEQ_MatVec(all_data,
                                all_data->matrix.P[fine_grid],
                                all_data->vector.u_coarse[coarse_grid],
                                all_data->vector.e[fine_grid]);
                     SEQ_Residual(all_data,
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
                  }
               }
               all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;

               this_grid = level;
               for (int i = 0; i < all_data->grid.n[this_grid]; i++){
                  all_data->vector.e[this_grid][i] = all_data->vector.u_fine[this_grid][i];
               }
               if (level > 0){
                  for (int inner_level = level; inner_level > 0; inner_level--){
                     fine_grid = inner_level - 1;
                     coarse_grid = inner_level;
                     prolong_start = omp_get_wtime();
                     SEQ_MatVec(all_data,
                                all_data->matrix.P[fine_grid],
                                all_data->vector.e[coarse_grid],
                                all_data->vector.e[fine_grid]);
                     all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
                  }
               }
               fine_grid = 0;
               for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                  e_write[level][i] = all_data->vector.e[fine_grid][i];
               }
            }
         }

         random_shuffle(level_perm.begin(), level_perm.end());

         for (int i = 0; i < all_data->grid.num_levels; i++){
            int level = level_perm[i];
            if (correct_flags[level] == 1){
               fine_grid = 0;
               vector<double> temp(all_data->grid.n[fine_grid]);
               for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                  all_data->vector.u[fine_grid][i] += e_write[level][i];
                  temp[i] = all_data->vector.u[fine_grid][i];
                  if (all_data->input.async_type == SEMI_ASYNC){
                     read[level][i] = all_data->vector.u[fine_grid][i];
                  }
               }
               read_hist.insert(read_hist.end(), temp.begin(), temp.end());

               double grid_wait =
                  (double)(all_data->grid.global_num_correct - all_data->grid.last_read_correct[level]);
               all_data->grid.mean_grid_wait[level] += grid_wait;
               all_data->grid.min_grid_wait[level] =
                  fmin(grid_wait, all_data->grid.min_grid_wait[level]);
               all_data->grid.max_grid_wait[level] =
                  fmax(grid_wait, all_data->grid.max_grid_wait[level]);

               if (all_data->input.print_grid_wait_flag == 1){
                  all_data->grid.grid_wait_hist.push_back((int)grid_wait);
               }

               all_data->grid.local_num_correct[level]++;
               all_data->grid.local_cycle_num_correct[level]++;
               all_data->grid.global_num_correct++;
               all_data->grid.last_read_correct[level] = all_data->grid.global_num_correct;
            }
         }
         all_data->output.sim_time_instance++;

         for (int level = 0; level < all_data->grid.num_levels; level++){
            if (grid_time_count[level] < grid_wait_list[level]){
               grid_time_count[level] = grid_time_count[level] + 1;
            }
            else{
               grid_wait_list[level] = (int)round(RandDouble(0.0, all_data->input.sim_grid_wait));
               grid_time_count[level] = 0;
            }
         }

         int break_flag = 1;
         if (all_data->input.converge_test_type == LOCAL){
            for (int level = 0; level < all_data->grid.num_levels; level++){
               if (all_data->grid.local_num_correct[level] < all_data->input.num_cycles){
                  break_flag = 0;
                  break;
               }
            }
            if (break_flag == 1){
               break;
            }
         }
         else{
            for (int level = 0; level < all_data->grid.num_levels; level++){
               if (all_data->grid.local_cycle_num_correct[level] == 0){
                  break_flag = 0;
                  break;
               }
            }
            if (break_flag == 1){
               for (int level = 0; level < all_data->grid.num_levels; level++){
                  all_data->grid.local_cycle_num_correct[level] = 0;
               }
               break;
            }
         }
      }
      all_data->output.sim_cycle_time_instance = all_data->output.sim_time_instance;

      SEQ_Residual(all_data,
                   all_data->matrix.A[fine_grid],
                   all_data->vector.f[fine_grid],
                   all_data->vector.u[fine_grid],
                   all_data->vector.y[fine_grid],
                   all_data->vector.r[fine_grid]);
      all_data->output.r_norm2 =
         Norm2(all_data->vector.r[fine_grid], all_data->grid.n[fine_grid]);
      if (all_data->input.print_reshist_flag == 1){
         printf("%d\t%d\t%e\n",
                k+1, all_data->output.sim_cycle_time_instance, all_data->output.r_norm2/all_data->output.r0_norm2);
      }
   }
}

void SEQ_Add_Vcycle_SimRand(AllData *all_data)
{
   srand(time(NULL));
   int fine_grid, coarse_grid, this_grid;
   int tid = 0;
   double residual_start;
   double smooth_start;
   double restrict_start;
   double prolong_start; 

   vector<double> read_hist(all_data->grid.n[0],0);
   vector<vector<double>> e_write(all_data->grid.num_levels, vector<double>(all_data->grid.n[0]));
   vector<vector<int>> last_row_read(all_data->grid.num_levels, vector<int>(all_data->grid.n[0]));
   double **read = (double **)calloc(all_data->grid.num_levels, sizeof(double *));
   vector<int> grid_wait_list(all_data->grid.num_levels);
   vector<int> correct_flags(all_data->grid.num_levels);
   vector<int> level_perm(all_data->grid.num_levels);

   for (int level = 0; level < all_data->grid.num_levels; level++){
      read[level] = (double *)calloc(all_data->grid.n[fine_grid], sizeof(double));
      fine_grid = 0;
      grid_wait_list[level] = (int)round(RandDouble(0.0, all_data->input.sim_grid_wait));
      level_perm[level] = level;
      if (all_data->input.read_type == READ_RES){
         for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
            read_hist[i] = all_data->vector.r[fine_grid][i];
         }
      }
   }

   for (int k = 0; k < all_data->input.num_cycles; k++){
      for (int level = 0; level < all_data->grid.num_levels; level++){
         all_data->grid.zero_flags[level] = 1;
	 double rand_num = RandDouble(0.0, 1.0);
	 correct_flags[level] = 0;
	 if (rand_num < .5){
	    correct_flags[level] = 1;
	 }
         if (correct_flags[level] == 1){
            if (all_data->input.async_type == FULL_ASYNC){ 
               fine_grid = 0;
               for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                  int rand_low = max(0, k - all_data->input.sim_read_delay);
	          rand_low = max(rand_low, last_row_read[level][i]);
                  int rand_high = k;
                  int col = (int)round(RandDouble((double)rand_low, (double)rand_high)); 
		  last_row_read[level][i] = col;
                  col *= all_data->grid.n[fine_grid];
                  read[level][i] = read_hist[col+i];
               }
            }
	    else {
	       int rand_low = max(0, k - all_data->input.sim_read_delay);
	       rand_low = max(rand_low, all_data->grid.last_read_correct[level]);
               int rand_high = k;
               int col = (int)round(RandDouble((double)rand_low, (double)rand_high));
	       all_data->grid.last_read_correct[level] = col;
               col *= all_data->grid.n[0];
	       for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                  read[level][i] = read_hist[col+i];
	       }
	    }

            fine_grid = 0;
            residual_start = omp_get_wtime();
	    if (all_data->input.read_type == READ_RES){
	       for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
		  all_data->vector.r[fine_grid][i] = read[level][i];
	       }
	    }
	    else {
	       SEQ_Residual(all_data,
                            all_data->matrix.A[fine_grid],
                            all_data->vector.f[fine_grid],
                            read[level],
                            all_data->vector.y[fine_grid],
                            all_data->vector.r[fine_grid]);
	    }
            all_data->output.residual_wtime[tid] += omp_get_wtime() - residual_start; 
            int coarsest_level;
            if (all_data->input.solver == MULTADD ||
                all_data->input.solver == ASYNC_MULTADD){
               coarsest_level = level;
            }
            else{
               coarsest_level = level+1;
            }

            for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
               fine_grid = inner_level;
               coarse_grid = inner_level + 1;
               if (inner_level < all_data->grid.num_levels-1){
                  restrict_start = omp_get_wtime();
                  SEQ_MatVec(all_data,
                             all_data->matrix.R[fine_grid],
                             all_data->vector.r[fine_grid],
                             all_data->vector.r[coarse_grid]);
                  all_data->output.restrict_wtime[tid] += omp_get_wtime() - restrict_start;
               }
            }
            smooth_start = omp_get_wtime();
            if (level == all_data->grid.num_levels-1){
               this_grid = all_data->grid.num_levels-1;
              // PARDISO(all_data->pardiso.info.pt,
              //         &(all_data->pardiso.info.maxfct),
              //         &(all_data->pardiso.info.mnum),
              //         &(all_data->pardiso.info.mtype),
              //         &(all_data->pardiso.info.phase),
              //         &(all_data->pardiso.csr.n),
              //         all_data->pardiso.csr.a,
              //         all_data->pardiso.csr.ia,
              //         all_data->pardiso.csr.ja,
              //         &(all_data->pardiso.info.idum),
              //         &(all_data->pardiso.info.nrhs),
              //         all_data->pardiso.info.iparm,
              //         &(all_data->pardiso.info.msglvl),
              //         all_data->vector.r[this_grid],
              //         all_data->vector.u_fine[this_grid],
              //         &(all_data->pardiso.info.error));
            }
            else {
               fine_grid = level;
               coarse_grid = level + 1;

               if (all_data->input.solver == MULTADD ||
                   all_data->input.solver == ASYNC_MULTADD){
                  for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                     all_data->vector.u_fine[fine_grid][i] = 0;
                  }
                  SMEM_Smooth(all_data,
                              all_data->matrix.A[fine_grid],
                              all_data->vector.r[fine_grid],
                              all_data->vector.u_fine[fine_grid],
                              all_data->vector.u_fine_prev[fine_grid],
                              all_data->vector.y[fine_grid],
                              all_data->input.num_fine_smooth_sweeps,
                              fine_grid,
                              0, 0); 
               }
               else{
                  for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
                     all_data->vector.u_fine[fine_grid][i] = 0;
                  }
                  for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
                     all_data->vector.u_coarse[coarse_grid][i] = 0;
                  }
                  SMEM_Smooth(all_data,
                              all_data->matrix.A[coarse_grid],
                              all_data->vector.r[coarse_grid],
                              all_data->vector.u_coarse[coarse_grid],
                              all_data->vector.u_coarse_prev[coarse_grid],
                              all_data->vector.y[coarse_grid],
                              all_data->input.num_coarse_smooth_sweeps,
                              coarse_grid,
                              0, 0);
                  SEQ_MatVec(all_data,
                             all_data->matrix.P[fine_grid],
                             all_data->vector.u_coarse[coarse_grid],
                             all_data->vector.e[fine_grid]);
                  SEQ_Residual(all_data,
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
               }
            }
            all_data->output.smooth_wtime[tid] += omp_get_wtime() - smooth_start;

            this_grid = level;
            for (int i = 0; i < all_data->grid.n[this_grid]; i++){
               all_data->vector.e[this_grid][i] = all_data->vector.u_fine[this_grid][i];
            }
            if (level > 0){
               for (int inner_level = level; inner_level > 0; inner_level--){
                  fine_grid = inner_level - 1;
                  coarse_grid = inner_level;
                  prolong_start = omp_get_wtime();
                  SEQ_MatVec(all_data,
                             all_data->matrix.P[fine_grid],
                             all_data->vector.e[coarse_grid],
                             all_data->vector.e[fine_grid]);
                  all_data->output.prolong_wtime[tid] += omp_get_wtime() - prolong_start;
               }
            }
            fine_grid = 0;
            for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
               e_write[level][i] = all_data->vector.e[fine_grid][i];
            }
         }
      }

      random_shuffle(level_perm.begin(), level_perm.end());

      for (int i = 0; i < all_data->grid.num_levels; i++){
         int level = level_perm[i];
         if (correct_flags[level] == 1){
            fine_grid = 0;
            for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
               all_data->vector.u[fine_grid][i] += e_write[level][i];
            }

            double grid_wait =
               (double)(all_data->grid.global_num_correct - all_data->grid.last_read_correct[level]);
            all_data->grid.mean_grid_wait[level] += grid_wait;
            all_data->grid.min_grid_wait[level] =
               fmin(grid_wait, all_data->grid.min_grid_wait[level]);
            all_data->grid.max_grid_wait[level] =
               fmax(grid_wait, all_data->grid.max_grid_wait[level]);

            if (all_data->input.print_grid_wait_flag == 1){
               all_data->grid.grid_wait_hist.push_back((int)grid_wait);
            }

            all_data->grid.local_num_correct[level]++;
            all_data->grid.local_cycle_num_correct[level]++;
            all_data->grid.global_num_correct++;
         }
      }

      fine_grid = 0;
      SEQ_Residual(all_data,
                   all_data->matrix.A[fine_grid],
                   all_data->vector.f[fine_grid],
                   all_data->vector.u[fine_grid],
                   all_data->vector.y[fine_grid],
                   all_data->vector.r[fine_grid]);
      vector<double> temp(all_data->grid.n[fine_grid]);
      if (all_data->input.read_type == READ_RES){
	 for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
            temp[i] = all_data->vector.r[fine_grid][i];
         }
      }
      else {
         for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
            temp[i] = all_data->vector.u[fine_grid][i];
         }
      }
      read_hist.insert(read_hist.end(), temp.begin(), temp.end());

      all_data->output.sim_time_instance++;
      all_data->output.sim_cycle_time_instance = all_data->output.sim_time_instance;

      all_data->output.r_norm2 =
         Norm2(all_data->vector.r[fine_grid], all_data->grid.n[fine_grid]);
      if (all_data->input.print_reshist_flag == 1){
         printf("%d\t%d\t%e\n",
                k+1, all_data->output.sim_cycle_time_instance, all_data->output.r_norm2/all_data->output.r0_norm2);
      }
   }
}
