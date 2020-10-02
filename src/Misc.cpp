#include "Main.hpp"
#include "Misc.hpp"

using namespace std;

void PrintOutput(AllData all_data)
{
   int mean_smooth_sweeps;
   int mean_cycles;
   double mean_vec_wtime, min_vec_wtime, max_vec_wtime;
   double mean_A_matvec_wtime, min_A_matvec_wtime, max_A_matvec_wtime;
   double mean_smooth_wtime, min_smooth_wtime, max_smooth_wtime;
   double mean_restrict_wtime, min_restrict_wtime, max_restrict_wtime;
   double mean_residual_wtime, min_residual_wtime, max_residual_wtime;
   double mean_prolong_wtime, min_prolong_wtime, max_prolong_wtime;
   double mean_updates, min_updates, max_updates;
   double mean_grid_wait, min_grid_wait, max_grid_wait;

   char print_str[1000];

   int finest_level;
   if (all_data.input.res_compute_type == GLOBAL){
      finest_level = 1;
   }
   else{
      finest_level = 0;
   }

   if (all_data.input.async_flag == 1 && all_data.input.async_type == SEMI_ASYNC){
      for (int level = finest_level; level < all_data.grid.num_levels; level++){
         all_data.grid.mean_grid_wait[level] /= (double)all_data.grid.local_num_correct[level];
      }
   }
   else{
      for (int level = finest_level; level < all_data.grid.num_levels; level++){
         all_data.grid.mean_grid_wait[level] = 0;
      }
   }

   mean_smooth_sweeps = (double)SumInt(all_data.output.smooth_sweeps, all_data.input.num_threads)/(double)all_data.input.num_threads;

   mean_grid_wait = SumDbl(all_data.grid.mean_grid_wait, all_data.grid.num_levels)/((double)(all_data.grid.num_levels-finest_level));
   max_grid_wait = MaxDouble(&all_data.grid.max_grid_wait[finest_level], all_data.grid.num_levels);
   min_grid_wait = MinDouble(&all_data.grid.min_grid_wait[finest_level], all_data.grid.num_levels);

   if (all_data.input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX || all_data.input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
      mean_updates = (double)SumInt(all_data.grid.local_num_correct, all_data.input.num_threads)/((double)(all_data.input.num_threads));
   }
   else {
      mean_updates = (double)SumInt(all_data.grid.local_num_correct, all_data.grid.num_levels)/((double)(all_data.grid.num_levels-finest_level));
   }
   min_updates = (double)MinInt(all_data.grid.local_num_correct, all_data.input.num_threads);
   max_updates = (double)MaxInt(all_data.grid.local_num_correct, all_data.input.num_threads);

   mean_smooth_wtime = SumDbl(all_data.output.smooth_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   min_smooth_wtime = MinDouble(all_data.output.smooth_wtime, all_data.input.num_threads);
   max_smooth_wtime = MaxDouble(all_data.output.smooth_wtime, all_data.input.num_threads);

   mean_residual_wtime = SumDbl(all_data.output.residual_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   min_residual_wtime = MinDouble(all_data.output.residual_wtime, all_data.input.num_threads);
   max_residual_wtime = MaxDouble(all_data.output.residual_wtime, all_data.input.num_threads);

   if (all_data.input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
      mean_restrict_wtime = MeanDoubleNonZero(all_data.output.restrict_wtime, all_data.input.num_threads);
      min_restrict_wtime = MinDoubleNonZero(all_data.output.restrict_wtime, all_data.input.num_threads);
      max_restrict_wtime = MaxDoubleNonZero(all_data.output.restrict_wtime, all_data.input.num_threads);
   }
   else {
      mean_restrict_wtime = SumDbl(all_data.output.restrict_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
      min_restrict_wtime = MinDouble(all_data.output.restrict_wtime, all_data.input.num_threads);
      max_restrict_wtime = MaxDouble(all_data.output.restrict_wtime, all_data.input.num_threads);
   }

   if (all_data.input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
      mean_prolong_wtime = MeanDoubleNonZero(all_data.output.prolong_wtime, all_data.input.num_threads);
      min_prolong_wtime = MinDoubleNonZero(all_data.output.prolong_wtime, all_data.input.num_threads);
      max_prolong_wtime = MaxDoubleNonZero(all_data.output.prolong_wtime, all_data.input.num_threads);
   }
   else {
      mean_prolong_wtime = SumDbl(all_data.output.prolong_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
      min_prolong_wtime = MinDouble(all_data.output.prolong_wtime, all_data.input.num_threads);
      max_prolong_wtime = MaxDouble(all_data.output.prolong_wtime, all_data.input.num_threads);
   }

   mean_A_matvec_wtime = SumDbl(all_data.output.A_matvec_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   min_A_matvec_wtime = MinDouble(all_data.output.A_matvec_wtime, all_data.input.num_threads);
   max_A_matvec_wtime = MaxDouble(all_data.output.A_matvec_wtime, all_data.input.num_threads);

   mean_vec_wtime = SumDbl(all_data.output.vec_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   min_vec_wtime = MinDouble(all_data.output.vec_wtime, all_data.input.num_threads);
   max_vec_wtime = MaxDouble(all_data.output.vec_wtime, all_data.input.num_threads);

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
                              "\tupdates = %d\n"
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
                        "\tUpdates = %f, %f, %f\n"
		        "\tSmooth time = %e, %e, %e\n" 
			"\tResidual time = %e, %e, %e\n"
			"\tRestrict time = %e, %e, %e\n"
		        "\tProlong time = %e, %e, %e\n"
                        "\tA-matvec time = %e, %e, %e\n"
                        "\tVec op time = %e, %e, %e\n"
			"\tGrid wait = %e, %e, %e\n"
			"\tGrid wait / num levels = %e, %e, %e\n");
   }
   else{
      strcpy(print_str, 
             "%e %e %e "
             "%e %e "
             "%f %f %f "
             "%e %e %e "
             "%e %e %e "
             "%e %e %e "
             "%e %e %e "
             "%e %e %e "
             "%e %e %e "
             "%e %e %e "
             "%e %e %e ");
   }

   printf(print_str,
          all_data.output.prob_setup_wtime, all_data.output.hypre_setup_wtime, all_data.output.setup_wtime,
          all_data.output.r_norm2/all_data.output.r0_norm2, all_data.output.solve_wtime,
          mean_updates, min_updates, max_updates,
          mean_smooth_wtime, min_smooth_wtime, max_smooth_wtime,
          mean_residual_wtime, min_residual_wtime, max_residual_wtime,
          mean_restrict_wtime, min_restrict_wtime, max_restrict_wtime,
          mean_prolong_wtime, min_prolong_wtime, max_prolong_wtime,
          mean_A_matvec_wtime, min_A_matvec_wtime, max_A_matvec_wtime,
          mean_vec_wtime, min_vec_wtime, max_vec_wtime,
          mean_grid_wait, min_grid_wait, max_grid_wait,
          mean_grid_wait/(double)all_data.grid.num_levels, min_grid_wait/(double)all_data.grid.num_levels, max_grid_wait/(double)all_data.grid.num_levels);

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

int MinInt(int *x, int n)
{
   int min_val = x[0];
   for (int i = 1; i < n; i++){
      if (x[i] < min_val){
         min_val = x[i];
      }
   }
   return min_val;
}

int MaxInt(int *x, int n)
{
   int max_val = x[0];
   for (int i = 1; i < n; i++){
      if (x[i] > max_val){
         max_val = x[i];
      }
   }
   return max_val;
}

double MinDoubleNonZero(double *x, int n)
{
   double min_val = DBL_MAX;
   for(int i = 0; i < n; i++){
      if(x[i] < min_val && x[i] > 0.0){
         min_val = x[i];
      }
   }
   return min_val;
}

double MaxDoubleNonZero(double *x, int n)
{
   double max_val = 0.0;
   for(int i = 0; i < n; i++){
      if(x[i] > max_val && x[i] > 0.0){
         max_val = x[i];
      }
   }
   return max_val;
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

double Parfor_InnerProd(double *x, int n)
{
   double sum = 0;
   #pragma omp parallel for reduction(+:sum)
   for (int i = 0; i < n; i++){
      sum += pow(x[i], 2.0);
   }
   return sum;
}

double Parfor_Norm2(double *x, int n)
{
   return sqrt(Parfor_InnerProd(x, n));
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

double MeanDoubleNonZero(double *x, int n)
{
   double sum = 0;
   double count = 0;
   for (int i = 0; i < n; i++){
      if (x[i] > 0.0){
         sum += x[i];
         count++;
      }
   }
   return sum/count;
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
      int finest_level;
      if (all_data->input.res_compute_type == GLOBAL){
         finest_level = 1;
      }
      else{
         finest_level = 0;
      }
      for (int level = finest_level; level < all_data->grid.num_levels; level++){
         if (all_data->grid.local_num_correct[level] < all_data->input.num_cycles){
            return 0;
         }
      }
      return 1;
   }
   return 0;
}

void SMEM_Barrier(AllData *all_data,
                 int *barrier_flags)
{
   int root = 0;
   int tid = omp_get_thread_num();
   int num_threads = omp_get_num_threads();

   barrier_flags[tid] = 1;
   #pragma omp flush (barrier_flags)
   while (1){
      if (tid == root){
         int s = 0;
        // #pragma omp flush (barrier_flags)
         for (int t = 0; t < num_threads; t++){
            s += barrier_flags[t];
         }
         if (s == num_threads){
            for (int t = 0; t < num_threads; t++){
               barrier_flags[t] = num_threads;
            }
           // #pragma omp flush (barrier_flags)
            return;
         }
      }
      else{
        // #pragma omp flush (barrier_flags)
         if (barrier_flags[tid] == num_threads){
            return;
         }
      }
   }
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

int SMEM_SRCLevelBarrier(AllData *all_data,
		         int *flag,
                         int level)
{
   int tid = omp_get_thread_num();
   if (all_data->barrier.local_sense[tid] == 0){
      all_data->barrier.local_sense[tid] = 1;
   }
   else{
      all_data->barrier.local_sense[tid] = 0;
   }
   omp_set_lock(&(all_data->barrier.lock));
   all_data->barrier.counter++;
   int arrived = all_data->barrier.counter;
   if (arrived == all_data->thread.level_threads[level].size()){
     // int read_converge_flag = all_data->thread.converge_flag;
      omp_unset_lock(&(all_data->barrier.lock));
      all_data->barrier.counter = 0;
      *flag = all_data->barrier.local_sense[tid];
      #pragma omp flush(flag)
   }
   else{
      omp_unset_lock(&(all_data->barrier.lock));
      while (*flag != all_data->barrier.local_sense[tid]){
         #pragma omp flush(flag)
      }
   }
   return 0;
}

void InitVectors(AllData *all_data)
{
   if (all_data->input.solver != EXPLICIT_EXTENDED_SYSTEM_BPX){
      if (all_data->input.thread_part_type == ALL_LEVELS/* &&
          all_data->input.num_threads > 1*/){
         for (int level = 0; level < all_data->grid.num_levels; level++){
            int coarsest_level;
            if (all_data->input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
               coarsest_level = all_data->grid.num_levels;
            }
            else {
               coarsest_level = level+2;
            }
            for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
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
                     all_data->level_vector[level].z[inner_level][i] = 0;
                     all_data->level_vector[level].z1[inner_level][i] = 0;
                     all_data->level_vector[level].z2[inner_level][i] = 0;
                  }
               }
            }
            int n = all_data->grid.n[level];
            if (all_data->input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
               for (int i = 0; i < n; i++){
                  all_data->vector.r[level][i] = 0;
                  all_data->vector.u[level][i] = 0;
                  all_data->vector.y[level][i] = 0;
                  all_data->vector.u_prev[level][i] = 0;
                  all_data->vector.e[level][i] = 0;
                  all_data->vector.z[level][i] = 0;
               }
            }
            else {
               if (level == 0){
                  for (int i = 0; i < n; i++){
                     all_data->vector.r[level][i] = 0;
                     all_data->vector.u[level][i] = 0;
                     all_data->vector.y[level][i] = 0;
                  }
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
   else {
      int n = hypre_CSRMatrixNumRows(all_data->matrix.AA);
      for (int i = 0; i < n; i++){
         all_data->vector.xx[i] = 0;
         all_data->vector.yy[i] = 0;
      }
      hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
      hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
      hypre_ParVectorSetConstantValues(U_array[0], 0.0);
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
 
   all_data->output.solve_wtime = 0;
   for (int t = 0; t < all_data->input.num_threads; t++){
      all_data->output.smooth_wtime[t] = 0;
      all_data->output.residual_wtime[t] = 0;
      all_data->output.restrict_wtime[t] = 0;
      all_data->output.prolong_wtime[t] = 0;
      all_data->output.A_matvec_wtime[t] = 0;
      all_data->output.vec_wtime[t] = 0;
      all_data->output.innerprod_wtime[t] = 0;
   }

   if (all_data->input.print_grid_wait_flag == 1){
      all_data->grid.grid_wait_hist.resize(0);
   }
   all_data->thread.converge_flag = 0;
}

vector<int> Divisors(int x)
{
   vector <int> divs;
   for (int i = 1; i <= x; i++){
      if (x % i == 0){
         divs.push_back(i);
      } 
   }
   return divs;
}

void FreeOrdering(OrderingData *P)
{
   free(P->dispv);
   free(P->disp);
   free(P->part);
}

void FreeMetis(MetisGraph *G)
{
   free(G->adjncy);
   free(G->adjwgt);
   free(G->xadj);
}

void FreeCSR(CSR *A)
{
   free(A->val);
   free(A->i);
   free(A->j_ptr);
}

void FreeTriplet(Triplet *T)
{
   free(T->i);
   free(T->j);
   free(T->val);
}

void WriteCSR(CSR A, char *out_str, int base)
{
   int row, col, k;
   double elem;
   char buffer[100];
   strcpy(buffer, out_str);
   FILE *out_file;
   remove(buffer);
   out_file = fopen(buffer, "a");
   for (int i = 0; i < A.n; i++){
      for (int j = A.j_ptr[i]; j < A.j_ptr[i+1]; j++){
         row = i;
         col = A.i[j];
         elem = A.val[j];
         fprintf(out_file, "%d   %d   %e\n", col+base, row+base, elem);
      }
   }
   fclose(out_file);
}

void PrintCSRMatrix(hypre_CSRMatrix *A, char *filename, int bin_file)
{
   HYPRE_Real *A_data;
   HYPRE_Int *A_i, *A_j;
   FILE *file_ptr;
   char buffer[100];

   int num_rows = hypre_CSRMatrixNumRows(A);
   int num_cols = hypre_CSRMatrixNumCols(A);
   int nnz = hypre_CSRMatrixNumNonzeros(A);

   sprintf(buffer, "%s", filename);

   if (bin_file){
      file_ptr = fopen(buffer, "wb");
      fwrite(&num_rows, sizeof(int), 1, file_ptr);
      fwrite(&num_cols, sizeof(int), 1, file_ptr);
      fwrite(&nnz, sizeof(double), 1, file_ptr);
   }
   else {
      file_ptr = fopen(buffer, "w");
      fprintf(file_ptr, "%d %d %d\n", num_rows, num_cols, nnz);
   }


   A_data = hypre_CSRMatrixData(A);
   A_i = hypre_CSRMatrixI(A);
   A_j = hypre_CSRMatrixJ(A);
   for (HYPRE_Int i = 0; i < num_rows; i++){
      for (HYPRE_Int jj = A_i[i]; jj < A_i[i+1]; jj++){
         HYPRE_Int ii = A_j[jj];
         int row = i+1;
         int col = ii+1;
         double elem = A_data[jj];
         if (bin_file == 1){
            fwrite(&row, sizeof(int), 1, file_ptr);
            fwrite(&col, sizeof(int), 1, file_ptr);
            fwrite(&elem, sizeof(double), 1, file_ptr);
         }
         else {
            fprintf(file_ptr, "%d %d %.16e\n", row, col, elem);
         }
      }
   }
   fclose(file_ptr);
}

void ReadBinary_fread_HypreParCSR(FILE *mat_file,
                                  hypre_ParCSRMatrix **A_ptr,
                                  int symm_flag,
                                  int include_disconnected_points_flag)
{
   using namespace std;
   size_t size;
   int temp_size;
   int k, q;
   int row, col;
   double elem;
   Triplet_AOS *buffer;

   fseek(mat_file, 0, SEEK_END);
   size = ftell(mat_file);
   rewind(mat_file);
   buffer = (Triplet_AOS *)malloc(sizeof(Triplet_AOS) * size);
   fread(buffer, sizeof(Triplet_AOS), size, mat_file);

   int file_lines = size/sizeof(Triplet_AOS);
   int num_rows = (int)buffer[0].i;
   int nnz = 0;

   vector<int> row_count(num_rows, 0);
   vector<int> col_count(num_rows, 0);
   vector<int> line_flag(file_lines, 0);

   for (int k = 1; k < file_lines; k++){
      row = buffer[k].i;
      col = buffer[k].j;
      elem = buffer[k].val;

      if (fabs(elem) > 0){
         col_count[col-1]++;
         row_count[row-1]++;
         line_flag[k] = 1;
         nnz++;
         if (symm_flag == 1 && row != col){
            nnz++;
         }
      }
   }

   // TODO: fix this
   int num_rows_old = num_rows;
   vector<int> flags_prefix_sum(num_rows, 0);
   if (include_disconnected_points_flag == 0){
      vector<int> disconnected_point_flag(num_rows, 0);
      int num_disconnected_points = 0;
      for (int k = 1; k < file_lines; k++){
         row = buffer[k].i;
         col = buffer[k].j;
         elem = buffer[k].val;

         if (line_flag[k] == 1){
            if (col_count[row-1] <= 1){
               line_flag[k] = 0;
               disconnected_point_flag[row-1] = 1;
               num_disconnected_points++;
               num_rows--;
               nnz--;
            }
           // if (row_count[col-1] <= 1){
           //    line_flag[k] = 0;
           // }
         }
      }
      partial_sum(disconnected_point_flag.begin(), disconnected_point_flag.end(), flags_prefix_sum.begin(), plus<int>());
   }

   vector<vector<int>> col_vec(num_rows);
   vector<vector<double>> elem_vec(num_rows);
   for (int k = 1; k < file_lines; k++){
      if (line_flag[k] == 1){
         row = buffer[k].i;
         col = buffer[k].j;
         elem = buffer[k].val;

         row -= flags_prefix_sum[row-1];
         col -= flags_prefix_sum[col-1];

         col_vec[row-1].push_back(col-1);
         elem_vec[row-1].push_back(elem);
         if (symm_flag == 1 && row != col){
            col_vec[col-1].push_back(row-1);
            elem_vec[col-1].push_back(elem);
         }
      }
   }

   HYPRE_IJMatrix Aij;
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, num_rows-1, 0, num_rows-1, &Aij);
   HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(Aij);

   for (int i = 0; i < num_rows; i++){
      int nnz = col_vec[i].size();
      double *values = (double *)malloc(nnz * sizeof(double));
      int *cols = (int *)malloc(nnz * sizeof(int));

      for (int j = 0; j < nnz; j++){
         cols[j] = col_vec[i][j];
         values[j] = elem_vec[i][j];
      }

      HYPRE_IJMatrixSetValues(Aij, 1, &nnz, &i, cols, values);

      free(values);
      free(cols);
   }

   HYPRE_IJMatrixAssemble(Aij);
   void *object;
   HYPRE_IJMatrixGetObject(Aij, &object);
   *A_ptr = (hypre_ParCSRMatrix *)object;
}
