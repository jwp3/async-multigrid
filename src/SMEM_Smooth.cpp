#include "Main.hpp"
#include "Misc.hpp"

void SMEM_Sync_Parfor_Jacobi(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *f,
                             HYPRE_Real *u,
                             HYPRE_Real *u_prev,
                             int num_sweeps,
                             int level)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   int tid = omp_get_thread_num();
   double smooth_weight = all_data->input.smooth_weight;

   for (int k = 0; k < num_sweeps; k++){
      #pragma omp for
     // if (tid == all_data->thread.barrier_root[level]){
      for (int i = 0; i < n; i++) u_prev[i] = u[i];
     // }
     // SMEM_LevelBarrier(all_data, level);
     // #pragma omp barrier
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

void SMEM_SemiAsync_Parfor_GaussSeidel(AllData *all_data,
                                      hypre_CSRMatrix *A,
                                      HYPRE_Real *f,
                                      HYPRE_Real *u,
                                      int num_sweeps)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   double smooth_weight = all_data->input.smooth_weight;

   for (int k = 0; k < num_sweeps; k++){
      #pragma omp for
      for (int i = 0; i < n; i++){
         if (A->data[A->i[i]] != 0.0)
         {
            res = f[i];
            for (int jj = A->i[i]; jj < A->i[i+1]; jj++)
            {
               ii = A->j[jj];
               res -= A->data[jj] * u[ii];
            }
            u[i] += smooth_weight * res / A->data[A->i[i]];
         }
      }
   }
}

void SMEM_Sync_Parfor_HybridJacobiGaussSeidel(AllData *all_data,
                                              hypre_CSRMatrix *A,
                                              HYPRE_Real *f,
                                              HYPRE_Real *u,
                                              HYPRE_Real *u_prev,
                                              int num_sweeps,
                                              int level)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);
   int t = omp_get_thread_num();
   int ns = all_data->thread.A_ns[level][t];
   int ne = all_data->thread.A_ne[level][t];

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

void SMEM_Sync_Jacobi(AllData *all_data,
                      hypre_CSRMatrix *A,
                      HYPRE_Real *f,
                      HYPRE_Real *u,
                      HYPRE_Real *u_prev,
                      int num_sweeps,
                      int thread_level,
                      int ns, int ne)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);
   double smooth_weight = all_data->input.smooth_weight;

   for (int k = 0; k < num_sweeps; k++){
      for (int i = ns; i < ne; i++) u_prev[i] = u[i];
      SMEM_LevelBarrier(all_data, thread_level);
      for (int i = ns; i < ne; i++){
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
      SMEM_LevelBarrier(all_data, thread_level);
   }
}

void SMEM_SemiAsync_GaussSeidel(AllData *all_data,
                                hypre_CSRMatrix *A,
                                HYPRE_Real *f,
                                HYPRE_Real *u,
                                int num_sweeps,
                                int thread_level,
                                int ns, int ne)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   for (int k = 0; k < num_sweeps; k++){
      for (int i = ns; i < ne; i++){
         if (A->data[A->i[i]] != 0.0)
         {
            res = f[i];
            for (int jj = A->i[i]; jj < A->i[i+1]; jj++)
            {
               ii = A->j[jj];
               res -= A->data[jj] * u[ii];
            }
            u[i] += res / A->data[A->i[i]];
         }
      }
      SMEM_LevelBarrier(all_data, thread_level);
   }
}


void SMEM_Sync_HybridJacobiGaussSeidel(AllData *all_data,
                                       hypre_CSRMatrix *A,
                                       HYPRE_Real *f,
                                       HYPRE_Real *u,
                                       HYPRE_Real *u_prev,
                                       int num_sweeps,
                                       int thread_level,
                                       int ns, int ne)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   for (int k = 0; k < num_sweeps; k++){
      for (int i = ns; i < ne; i++) u_prev[i] = u[i];
      SMEM_LevelBarrier(all_data, thread_level);
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
      SMEM_LevelBarrier(all_data, thread_level);
   }
}
