#include "Main.hpp"
#include "Misc.hpp"
#include "SMEM_MatVec.hpp"
#include "SMEM_Smooth.hpp"

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
      if (k == 0 && all_data->vector.zero_flag == 1){
         #pragma omp for
         for (int i = 0; i < n; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               u[i] += smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      else{
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
}

void SMEM_Sync_Parfor_L1Jacobi(AllData *all_data,
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

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->vector.zero_flag == 1){
         #pragma omp for
         for (int i = 0; i < n; i++){
            res = f[i];
            u[i] += res / all_data->matrix.L1_row_norm[level][i];
         }
      }
      else{
         #pragma omp for
         for (int i = 0; i < n; i++) u_prev[i] = u[i];
         #pragma omp for
         for (int i = 0; i < n; i++){
            res = f[i];
            for (int jj = A->i[i]; jj < A->i[i+1]; jj++)
            {
               ii = A->j[jj];
               res -= A->data[jj] * u_prev[ii];
            }
            u[i] += res / all_data->matrix.L1_row_norm[level][i];
         }
      }
   }
}

void SMEM_SemiAsync_Parfor_GaussSeidel(AllData *all_data,
                                      hypre_CSRMatrix *A,
                                      HYPRE_Real *f,
                                      HYPRE_Real *u,
                                      int num_sweeps,
                                      int level)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

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
            u[i] += res / A->data[A->i[i]];
         }
      }
   }
}

void SMEM_Async_Parfor_GaussSeidel(AllData *all_data,
                                   hypre_CSRMatrix *A,
                                   HYPRE_Real *f,
                                   HYPRE_Real *u,
                                   int num_sweeps,
                                   int level)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   for (int k = 0; k < num_sweeps; k++){
      #pragma omp for nowait
      for (int i = 0; i < n; i++){
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
      if (k == 0 && all_data->vector.zero_flag == 1){
         for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
                  ii = A->j[jj];
                  if (ii >= ns && ii < ne){
                     res -= A->data[jj] * u[ii];
                  }
               }
               u[i] += res / A->data[A->i[i]];
            }
         }
      }
      else{
         #pragma omp for
         for (int i = 0; i < n; i++) u_prev[i] = u[i]; 
         for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
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
      if (k == 0 && all_data->vector.zero_flag == 1){
         for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               u[i] += smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      else{
         for (int i = ns; i < ne; i++) u_prev[i] = u[i];
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
         for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
                  ii = A->j[jj];
                  res -= A->data[jj] * u_prev[ii];
               }
               u[i] += smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
   }
}

void SMEM_Sync_L1Jacobi(AllData *all_data,
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
      if (k == 0 && all_data->vector.zero_flag == 1){
         for (int i = ns; i < ne; i++){
            res = f[i];
            u[i] += res / all_data->matrix.L1_row_norm[thread_level][i];
         }
      }
      else{
         for (int i = ns; i < ne; i++) u_prev[i] = u[i];
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
         for (int i = ns; i < ne; i++){
            res = f[i];
            for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
               ii = A->j[jj];
               res -= A->data[jj] * u_prev[ii];
            }
            u[i] += res / all_data->matrix.L1_row_norm[thread_level][i];
         }
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
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
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
   }
}

void SMEM_Async_GaussSeidel(AllData *all_data,
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
         if (A->data[A->i[i]] != 0.0){
            res = f[i];
            for (int jj = A->i[i]; jj < A->i[i+1]; jj++)
            {
               ii = A->j[jj];
               res -= A->data[jj] * u[ii];
            }
            u[i] += res / A->data[A->i[i]];
         }
      }
   }
   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
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
      if (k == 0 && all_data->vector.zero_flag == 1){
         for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
                  ii = A->j[jj];
                  if (ii >= ns && ii < ne){
                     res -= A->data[jj] * u[ii];
                  }
               }
               u[i] += res / A->data[A->i[i]];
            }
         }
      }
      else{
         for (int i = ns; i < ne; i++) u_prev[i] = u[i];
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
         for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
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
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
   }
}

void SMEM_Sync_SymmetricJacobi(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *f,
                               HYPRE_Real *u,
                               HYPRE_Real *y,
                               HYPRE_Real *r,
                               int num_sweeps,
                               int thread_level,
                               int ns, int ne)
{
   int k = 0;
   for (int i = ns; i < ne; i++){
      r[i] = f[i];
   }
   while(1){
      for (int i = ns; i < ne; i++){
         if (A->data[A->i[i]] != 0.0){
            r[i] *= all_data->input.smooth_weight / A->data[A->i[i]];
         }
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);

      SMEM_MatVec(all_data, A, r, y, ns, ne);
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);

      for (int i = ns; i < ne; i++){
         if (A->data[A->i[i]] != 0.0){
            r[i] = (2.0 * A->data[A->i[i]] * r[i] / all_data->input.smooth_weight) - y[i];
            r[i] *= all_data->input.smooth_weight / A->data[A->i[i]];
         }
         u[i] += r[i];
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
      k++;
      if (k == num_sweeps){
         break;
      }
      SMEM_Residual(all_data, A, f, u, y, r, ns, ne);
   }
}

void SMEM_Sync_SymmetricL1Jacobi(AllData *all_data,
                                 hypre_CSRMatrix *A,
                                 HYPRE_Real *f,
                                 HYPRE_Real *u,
                                 HYPRE_Real *y,
                                 HYPRE_Real *r,
                                 int num_sweeps,
                                 int thread_level,
                                 int ns, int ne)
{
   int k = 0;
   for (int i = ns; i < ne; i++){
      r[i] = f[i];
   }
   while(1){
      for (int i = ns; i < ne; i++){
         r[i] /= all_data->matrix.L1_row_norm[thread_level][i];
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);

      SMEM_MatVec(all_data, A, r, y, ns, ne);
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);

      for (int i = ns; i < ne; i++){
         r[i] = (2.0 * all_data->matrix.L1_row_norm[thread_level][i] * r[i]) - y[i];
         r[i] /= all_data->matrix.L1_row_norm[thread_level][i];
         u[i] += r[i];
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
      k++;
      if (k == num_sweeps){
         break;
      }
      SMEM_Residual(all_data, A, f, u, y, r, ns, ne);
   }
}
