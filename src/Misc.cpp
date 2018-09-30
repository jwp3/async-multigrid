#include "Main.hpp"
#include "Misc.hpp"

void PrintOutput(AllData all_data)
{
   int mean_smooth_sweeps;
   int mean_cycles;
   double mean_smooth_wtime;
   double mean_restrict_wtime;
   double mean_residual_wtime;
   double mean_prolong_wtime;
   double mean_correct;
   double mean_grid_wait;
   double max_grid_wait;
   double min_grid_wait;

   char print_str[1000];

   if (all_data.input.async_flag == 1){
      for (int level = 0; level < all_data.grid.num_levels; level++){
         all_data.grid.mean_grid_wait[level] /= (double)all_data.grid.local_num_correct[level];
      }
   }
   else{
      for (int level = 0; level < all_data.grid.num_levels; level++){
         all_data.grid.min_grid_wait[level] = 0;
      }
   }

   mean_grid_wait =
      SumDbl(all_data.grid.mean_grid_wait, all_data.grid.num_levels)/(double)all_data.grid.num_levels;
   max_grid_wait =
      MaxDouble(all_data.grid.max_grid_wait, all_data.grid.num_levels);
   min_grid_wait =
      MinDouble(all_data.grid.min_grid_wait, all_data.grid.num_levels);

   mean_smooth_sweeps =
      (int)((double)SumInt(all_data.output.smooth_sweeps, all_data.input.num_threads)/(double)all_data.input.num_threads);
   mean_correct =
      (int)((double)SumInt(all_data.grid.local_num_correct, all_data.grid.num_levels)/(double)all_data.grid.num_levels);
   mean_smooth_wtime =
      SumDbl(all_data.output.smooth_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_residual_wtime =
      SumDbl(all_data.output.residual_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_restrict_wtime =
      SumDbl(all_data.output.restrict_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_prolong_wtime =
      SumDbl(all_data.output.prolong_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;

   if (all_data.input.print_level_stats_flag == 1){
      for (int level = 0; level < all_data.grid.num_levels; level++){
         int root = all_data.thread.barrier_root[level];
	 double level_mean_smooth_wtime =
            SumDbl(&all_data.output.smooth_wtime[root], all_data.thread.level_threads[level].size())/
		(double)all_data.thread.level_threads[level].size();
         double level_mean_residual_wtime =
            SumDbl(&all_data.output.residual_wtime[root], all_data.thread.level_threads[level].size())/
		(double)all_data.thread.level_threads[level].size();
         double level_mean_restrict_wtime =
            SumDbl(&all_data.output.restrict_wtime[root], all_data.thread.level_threads[level].size())/
		(double)all_data.thread.level_threads[level].size();
         double level_mean_prolong_wtime =
            SumDbl(&all_data.output.prolong_wtime[root], all_data.thread.level_threads[level].size())/
		(double)all_data.thread.level_threads[level].size();
         if (all_data.input.format_output_flag == 0){
            strcpy(print_str, "Level %d stats:\n"
                              "\tcorrections = %d\n"
                              "\tsmooth time = %e\n"
                              "\tresidual time = %e\n"
                              "\trestrict time = %e\n"
                              "\tprolong time = %e\n"
                              "\tMean grid wait = %e\n"
                              "\tMax grid wait = %e\n"
                              "\tMin grid wait = %e\n");
         }
         else{
            strcpy(print_str, "%d %d %e %e %e %e %e %e %e\n");
         }

         printf(print_str,
                level,
                all_data.grid.local_num_correct[level],
                level_mean_smooth_wtime,
                level_mean_residual_wtime,
                level_mean_restrict_wtime,
                level_mean_prolong_wtime,
                all_data.grid.mean_grid_wait[level]/(double)all_data.grid.num_levels,
                all_data.grid.max_grid_wait[level]/(double)all_data.grid.num_levels,
                all_data.grid.min_grid_wait[level]/(double)all_data.grid.num_levels);
      }
   }

   if (all_data.input.format_output_flag == 0){
      strcpy(print_str, "Setup stats:\n"
			"\tProblem setup time = %e\n"
                        "\tHypre setup time = %e\n"
                        "\tRemaining setup time = %e\n"
                        "Solve stats:\n"
                        "\tRelative Residual 2-norm = %e\n"
                        "\tTotal solve time = %e\n"
                        "\tMean corrections = %f\n"
		        "\tMean smooth time = %e\n" 
			"\tMean residual time = %e\n"
			"\tMean restrict time = %e\n"
		        "\tMean prolong time = %e\n"
			"\tMean computation time = %e\n"
			"\tMean grid wait = %e\n"
			"\tMax grid wait = %e\n"
                        "\tMin grid wait = %e\n"
			"\tMean grid wait / num levels = %e\n"
                        "\tMax grid wait / num levels = %e\n"
                        "\tMin grid wait / num levels = %e\n");
   }
   else{
      strcpy(print_str, "%e %e %e %e %e %f %e %e %e %e %e %e %e %e %e %e %e ");
   }

   double mean_comp_time = mean_smooth_wtime + 
			   mean_residual_wtime +
			   mean_restrict_wtime +
			   mean_prolong_wtime;
   printf(print_str,
          all_data.output.prob_setup_wtime,
          all_data.output.hypre_setup_wtime,
          all_data.output.setup_wtime,
          all_data.output.r_norm2/all_data.output.r0_norm2,
          all_data.output.solve_wtime,
          mean_correct,
          mean_smooth_wtime,
          mean_residual_wtime,
          mean_restrict_wtime,
          mean_prolong_wtime,
	  mean_comp_time,
          mean_grid_wait,
          max_grid_wait,
          min_grid_wait,
          mean_grid_wait/(double)all_data.grid.num_levels,
          max_grid_wait/(double)all_data.grid.num_levels,
          min_grid_wait/(double)all_data.grid.num_levels);

   if (all_data.input.mfem_test_error_flag == 1){
      if (all_data.input.format_output_flag == 0){
         printf("\tmfem error = %e\n", all_data.output.mfem_e_norm2);
      }
      else{
         printf("%e ", all_data.output.mfem_e_norm2);
      }
   }
   if (all_data.input.hypre_test_error_flag == 1){
      if (all_data.input.format_output_flag == 0){
         printf("\thypre error = %e\n", all_data.output.hypre_e_norm2);
      }
      else{
         printf("%e ", all_data.output.hypre_e_norm2);
      }
   }
   if (all_data.input.format_output_flag == 1){
      printf("\n");
   }
   if (all_data.input.print_grid_wait_flag == 1){
      for (int i = 0; i < all_data.grid.grid_wait_hist.size(); i++){
         printf("%d\n", all_data.grid.grid_wait_hist[i]);
      }
   }
}

double MinDouble(double *x, int n)
{
   double min_val = x[0];
   for(int i = 1; i < n; i++){
      if(x[i] < min_val){
         min_val = x[i];
      }
   }
   return min_val;
}

double MaxDouble(double *x, int n)
{
   double max_val = x[0];
   for(int i = 1; i < n; i++){
      if(x[i] > max_val){
         max_val = x[i];
      }
   }
   return max_val;
}

double RandDouble(double low, double high)
{
   return low + (high - low) * ((double)rand() / RAND_MAX);
}

double Norm2(double *x, int n)
{
   double sum = 0;
   for (int i = 0; i < n; i++){
      sum += pow(x[i], 2.0);
   }
   return sqrt(sum);
}

double Parfor_Norm2(double *x, int n)
{
   double sum = 0;
   #pragma omp parallel for reduction(+:sum)
   for (int i = 0; i < n; i++){
      sum += pow(x[i], 2.0);
   }
   return sqrt(sum);
}

void Par_Norm2(AllData *all_data,
               double *r,
               int thread_level,
               int ns, int ne)
{
   int tid = omp_get_thread_num();
   all_data->thread.loc_sum[tid] = 0;
   for (int i = ns; i < ne; i++){
      all_data->thread.loc_sum[tid] += pow(r[i], 2.0);
   }
   #pragma omp atomic
   all_data->output.r_norm2 += all_data->thread.loc_sum[tid];
   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
   if (tid == all_data->thread.barrier_root[thread_level]){
      all_data->output.r_norm2 = sqrt(all_data->output.r_norm2);
   }
}


int SumInt(int *x, int n)
{
   int sum = 0;
   for (int i = 0; i < n; i++){
      sum += x[i];
   }
   return sum;
}

double SumDbl(double *x, int n)
{
   double sum = 0;
   for (int i = 0; i < n; i++){
     sum += x[i];
   }
   return sum;
}

void SwapInt(int *xp, int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void SwapDouble(double *xp, double *yp)
{
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void BubblesortPair_int_double(int *x, double *y, int n)
{
   for (int i = 0; i < n-1; i++){
       for (int j = 0; j < n-i-1; j++){
           if (x[j] > x[j+1]){
              SwapInt(&x[j], &x[j+1]);
              SwapDouble(&y[j], &y[j+1]);
           }
       }
   }
}

void QuicksortPair_int_double(int *x, double *y, int left, int right)
{
   int i = left, j = right+1, pivot = x[left], temp;
   double temp_double;
   if (left < right){
      while(1){
         do{
            ++i;
         }while((x[i] <= pivot) && (i <= right));
         do{
            --j;
         }while(x[j] > pivot);
         if (i >= j) break;
         temp = x[i];
         x[i] = x[j];
         x[j] = temp;
         temp_double = y[i];
         y[i] = y[j];
         y[j] = temp_double;
      }
      temp = x[left];
      x[left] = x[j];
      x[j] = temp;
      temp_double = y[left];
      y[left] = y[j];
      y[j] = temp_double;
      QuicksortPair_int_double(x, y, left, j-1);
      QuicksortPair_int_double(x, y, j+1, right);
   }
}

int CheckConverge(AllData *all_data,
                  int thread_level)
{
   if (all_data->input.check_resnorm_flag == 1){
      if (all_data->output.r_norm2/all_data->output.r0_norm2 < all_data->input.tol){
         return 1;
      }
   }
   if (all_data->input.converge_test_type == ALL_LEVELS){
      for (int q = 0; q < all_data->grid.num_levels; q++){
         if (all_data->grid.local_num_correct[q] < all_data->input.num_cycles){
            return 0;
         }
      }
      return 1;
   }
   return 0;
}

int SMEM_LevelBarrier(AllData *all_data,
                      int **barrier_flags,
                      int level)
{
   int root = all_data->thread.barrier_root[level];
   int tid = omp_get_thread_num();
   int t;

   barrier_flags[level][tid] = 1;
   #pragma omp flush (barrier_flags)
   while (1){
      if (tid == root){
         int s = 0;
        // #pragma omp flush (barrier_flags)
         for (int i = 0; i < all_data->thread.level_threads[level].size(); i++){
            t = all_data->thread.level_threads[level][i];
            s += barrier_flags[level][t];
         }
         if (s == all_data->thread.level_threads[level].size()){
            int read_converge_flag = all_data->thread.converge_flag;
            for (int i = 0; i < all_data->thread.level_threads[level].size(); i++){
               t = all_data->thread.level_threads[level][i];
               if (read_converge_flag == 1){
                  barrier_flags[level][t] = 2*all_data->thread.level_threads[level].size();
               }
               else{
                  barrier_flags[level][t] = all_data->thread.level_threads[level].size();
               }
            }
           // #pragma omp flush (barrier_flags)
            if (read_converge_flag == 1){
               return 1;
            }
            else{
               return 0;
            }
         }
      }
      else{
        // #pragma omp flush (barrier_flags)
         if (barrier_flags[level][tid] == 2*all_data->thread.level_threads[level].size()){
            return 1;
         }
         else if (barrier_flags[level][tid] == all_data->thread.level_threads[level].size()){
            return 0;
         }
      }
   }
}

void InitVectors(AllData *all_data)
{
   if (all_data->input.thread_part_type == ALL_LEVELS &&
       all_data->input.num_threads > 1){
      for (int level = 0; level < all_data->grid.num_levels; level++){
         for (int inner_level = 0; inner_level < level+2; inner_level++){
            if (inner_level < all_data->grid.num_levels){
               int n = all_data->grid.n[inner_level];
               for (int i = 0; i < n; i++){
                  all_data->level_vector[level].f[inner_level][i] = 0;
                  all_data->level_vector[level].u[inner_level][i] = 0;
                  all_data->level_vector[level].u_prev[inner_level][i] = 0;
                  all_data->level_vector[level].u_coarse[inner_level][i] = 0;
                  all_data->level_vector[level].u_coarse_prev[inner_level][i] = 0;
                  all_data->level_vector[level].u_fine[inner_level][i] = 0;
                  all_data->level_vector[level].u_fine_prev[inner_level][i] = 0;
                  all_data->level_vector[level].y[inner_level][i] = 0;
                  all_data->level_vector[level].r[inner_level][i] = 0;
                  all_data->level_vector[level].r_coarse[inner_level][i] = 0;
                  all_data->level_vector[level].r_fine[inner_level][i] = 0;
                  all_data->level_vector[level].e[inner_level][i] = 0;
               }
            }
         }
         if (level == 0){
            int n = all_data->grid.n[level];
            for (int i = 0; i < n; i++){
               all_data->vector.r[level][i] = 0;
               all_data->vector.u[level][i] = 0;
               all_data->vector.y[level][i] = 0;
            }
         }
      }
   }
   else {
      for (int level = 0; level < all_data->grid.num_levels; level++){
         int n = all_data->grid.n[level];
         for (int i = 0; i < n; i++){
            if (level > 0){
               all_data->vector.f[level][i] = 0;
            }
            all_data->vector.u[level][i] = 0;
            all_data->vector.u_prev[level][i] = 0;
            all_data->vector.u_coarse[level][i] = 0;
            all_data->vector.u_coarse_prev[level][i] = 0;
            all_data->vector.u_fine[level][i] = 0;
            all_data->vector.u_fine_prev[level][i] = 0;
            all_data->vector.y[level][i] = 0;
            all_data->vector.r[level][i] = 0;
            all_data->vector.r_coarse[level][i] = 0;
            all_data->vector.r_fine[level][i] = 0;
            all_data->vector.e[level][i] = 0;
         }
      }
   }
}

void InitSolve(AllData *all_data)
{
   InitVectors(all_data);
   all_data->output.sim_time_instance = 0;
   all_data->output.sim_cycle_time_instance = 0;
   all_data->grid.global_num_correct = 0;
   all_data->grid.global_cycle_num_correct = 0;
   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->grid.num_smooth_wait[level] = 0;
      all_data->grid.local_num_res_compute[level] = 0;
      all_data->grid.local_num_correct[level] = 0;
      all_data->grid.local_cycle_num_correct[level] = 0;
      all_data->grid.last_read_correct[level] = 0;
      all_data->grid.last_read_cycle_correct[level] = 0;
      all_data->grid.mean_grid_wait[level] = 0;
      all_data->grid.max_grid_wait[level] = 0;
      all_data->grid.min_grid_wait[level] = DBL_MAX;
   }
   all_data->thread.converge_flag = 0;

   all_data->output.solve_wtime = 0;
   for (int t = 0; t < all_data->input.num_threads; t++){
      all_data->output.smooth_wtime[t] = 0;
      all_data->output.residual_wtime[t] = 0;
      all_data->output.restrict_wtime[t] = 0;
      all_data->output.prolong_wtime[t] = 0;
   }

   if (all_data->input.print_grid_wait_flag == 1){
      all_data->grid.grid_wait_hist.resize(0);
   }
}
