#include "Main.hpp"

void SMEM_Sync_Parfor_Jacobi(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *f,
                             HYPRE_Real *u,
                             HYPRE_Real *u_prev,
                             int num_sweeps)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   double smooth_weight = all_data->input.smooth_weight;

   for (int k = 0; k < num_sweeps; k++){
      #pragma omp for
      for (int i = 0; i < n; i++) u_prev[i] = u[i];
      #pragma omp for
      for (int i = 0; i < n; i++){
         if (A->data[A->i[i]] != 0.0)
         {
            res = f[i];
            for (int jj = A->i[i]; jj < A->i[i+1]; jj++)
            {
               ii = A->j[jj];
               res -= A->data[jj] * u_prev[ii];
            }
            u[i] += smooth_weight * res / A->data[A->i[i]];
         }
      }
   }
}

void SMEM_Sync_HybridJacobiGaussSeidel(AllData *all_data,
                                       hypre_CSRMatrix *A,
                                       HYPRE_Real *f,
                                       HYPRE_Real *u,
                                       HYPRE_Real *u_prev,
                                       int num_sweeps)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);
   int num_threads = all_data->input.num_threads;
   int t = omp_get_thread_num();

   int ns, ne;
   int size = n/num_threads;
   int rest = n - size*num_threads;
   if (t < rest)
   {
      ns = t*size + t;
      ne = (t + 1)*size + t + 1;
   }
   else
   {
      ns = t*size + rest;
      ne = (t + 1)*size + rest;
   }

   for (int k = 0; k < num_sweeps; k++){
      #pragma omp for
      for (int i = 0; i < n; i++) u_prev[i] = u[i]; 
      for (int i = ns; i < ne; i++){
         if (A->data[A->i[i]] != 0.0)
         {
            res = f[i];
            for (int jj = A->i[i]; jj < A->i[i+1]; jj++)
            {
               ii = A->j[jj];
               if (ii >= ns && ii < ne){
                  res -= A->data[jj] * u[ii];
               }
               else{
                  res -= A->data[jj] * u_prev[ii];
               }
            }
            u[i] += res / A->data[A->i[i]];
         }
      }
      #pragma omp barrier
   }
}

