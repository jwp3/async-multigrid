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

void QuicksortPair_int_dbl(int *x, double *y, int left, int right)
{
   int i = left, j = right+1, pivot = x[left], temp;
   double temp_dbl;
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
         temp_dbl = y[i];
         y[i] = y[j];
         y[j] = temp_dbl;
      }
      temp = x[left];
      x[left] = x[j];
      x[j] = temp;
      temp_dbl = y[left];
      y[left] = y[j];
      y[j] = temp_dbl;
      QuicksortPair_int_dbl(x, y, left, j-1);
      QuicksortPair_int_dbl(x, y, j+1, right);
   }
}

/* untested barrier code */
void SMEM_Barrier(AllData *all_data,
                  int level)
{
   int root = all_data->thread.barrier_root;
   int tid = omp_get_thread_num();
   int t;

   all_data->thread.barrier_flags[tid] = 1;
   while (1){
      if (tid == root){
         int s = 0;
         for (int i = 0; i < all_data->thread.level_threads[level].size(); i++){
            t = all_data->thread.level_threads[level][i];
            s += all_data->thread.barrier_flags[t];
         }
         if (s == all_data->thread.level_threads[level].size() - 1){
            for (int i = 0; i < all_data->thread.level_threads[level].size(); i++){
               t = all_data->thread.level_threads[level][i];
               all_data->thread.barrier_flags[t] = all_data->thread.level_threads[level].size();
            }
            break;
         }
      }
      else{
         if (all_data->thread.barrier_flags[tid] == all_data->thread.level_threads[level].size()){
            break;
         }
      }
   }
   all_data->thread.barrier_flags[tid] = 0;
}
