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

   mean_smooth_sweeps =
      (int)((double)SumInt(all_data.output.smooth_sweeps, all_data.input.num_threads)/(double)all_data.input.num_threads);
   mean_cycles =
      (int)((double)SumInt(all_data.output.cycles, all_data.input.num_threads)/(double)all_data.input.num_threads);
   mean_smooth_wtime =
      SumDbl(all_data.output.smooth_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_residual_wtime =
      SumDbl(all_data.output.residual_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_restrict_wtime =
      SumDbl(all_data.output.restrict_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_prolong_wtime =
      SumDbl(all_data.output.prolong_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;

   char print_str[1000];
   if (all_data.input.format_output_flag == 0){
      strcpy(print_str, "\nSetup stats:\n"
                        "\tHypre setup time = %e\n"
                        "\tRemaining setup time = %e\n"
                        "\tTotal setup time = %e\n"
                        "\nSolve stats:\n"
                        "\tRelative Residual 2-norm = %e\n"
                        "\tTotal solve time = %e\n"
		        "\tMean smooth time = %e\n" 
			"\tMean residual time = %e\n"
			"\tMean restrict time = %e\n"
		        "\tMean prolong time = %e\n");
   }
   else{
      strcpy(print_str, "%e %e %e %e %e %e %e %e %e\n");
   }

   printf(print_str,
          all_data.output.hypre_setup_wtime,
          all_data.output.setup_wtime,
          all_data.output.hypre_setup_wtime + all_data.output.hypre_setup_wtime,
          all_data.output.r_norm2/all_data.output.r0_norm2,
          all_data.output.solve_wtime,
          mean_smooth_wtime,
          mean_residual_wtime,
          mean_restrict_wtime,
          mean_prolong_wtime);
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

int CheckConverge(AllData *all_data)
{
   for (int q = 0; q < all_data->grid.num_levels; q++){
      if (all_data->grid.num_correct[q] < all_data->input.num_cycles){
         return 0;
      }
   }
   return 1;
}

int SMEM_LevelBarrier(AllData *all_data,
                      int **barrier_flags,
                      int level)
{
   int root = all_data->thread.barrier_root[level];
   int tid = omp_get_thread_num();
   int t;

   barrier_flags[level][tid] = 1;
   while (1){
      if (tid == root){
         int s = 0;
         #pragma omp flush (barrier_flags)
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
            #pragma omp flush (barrier_flags)
            if (read_converge_flag == 1){
               return 1;
            }
            else{
               return 0;
            }
         }
      }
      else{
         #pragma omp flush (barrier_flags)
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
   if (all_data->input.thread_part_type == ALL_LEVELS){
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
            all_data->vector.f[level][i] = 0;
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
   int level = 0;
   for (int i = 0; i < all_data->grid.n[level]; i++){
      all_data->vector.f[level][i] = 1;
   }
}

void InitSolve(AllData *all_data)
{
   InitVectors(all_data);
   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->grid.num_correct[level] = 0;
   }
   all_data->thread.converge_flag = 0;

   all_data->output.solve_wtime = 0;
   for (int t = 0; t < all_data->input.num_threads; t++){
      all_data->output.smooth_wtime[t] = 0;
      all_data->output.residual_wtime[t] = 0;
      all_data->output.restrict_wtime[t] = 0;
      all_data->output.prolong_wtime[t] = 0;
   }
}
