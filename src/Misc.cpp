#include "Main.hpp"
#include "Misc.hpp"

void PrintOutput(AllData all_data)
{
   int mean_smooth_sweeps;
   int mean_cycles;
   double mean_smooth_wtime;
   double mean_restrict_wtime;
   double mean_resid_wtime;
   double mean_prolong_wtime;

   mean_smooth_sweeps =
      (int)((double)SumInt(all_data.output.smooth_sweeps, all_data.input.num_threads)/(double)all_data.input.num_threads);
   mean_cycles =
      (int)((double)SumInt(all_data.output.cycles, all_data.input.num_threads)/(double)all_data.input.num_threads);
   mean_smooth_wtime =
      SumDbl(all_data.output.smooth_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_resid_wtime =
      SumDbl(all_data.output.residual_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_restrict_wtime =
      SumDbl(all_data.output.restrict_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;
   mean_prolong_wtime =
      SumDbl(all_data.output.prolong_wtime, all_data.input.num_threads)/(double)all_data.input.num_threads;

   if (all_data.input.format_output_flag){
   }
   else {
      printf("relative residual 2-norm = %e\n"
             "number of multigrid cycles = %d\n"
	     "mean smoothing sweeps = %d\n"
             "total multigrid time = %e\n"
             "smoothing time = %e\n"
             "restriction time = %e\n"
             "prolongation time = %e\n"
             "residual computation time = %e\n",
             all_data.output.r_norm2/all_data.output.r0_norm2,
             mean_cycles,
             mean_smooth_sweeps,
             all_data.output.solve_wtime,
             mean_smooth_wtime,
             mean_restrict_wtime,
             mean_prolong_wtime,
             mean_resid_wtime);
   }
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
         else if(barrier_flags[level][tid] == all_data->thread.level_threads[level].size()){
            return 0;
         }
      }
   }
}
