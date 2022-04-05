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
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
         #pragma omp for
         for (int i = 0; i < n; i++){
            if (A->data[A->i[i]] != 0.0){
               u[i] = smooth_weight * f[i] / A->data[A->i[i]];
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

void SMEM_Async_Parfor_Jacobi(AllData *all_data,
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
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
	 #pragma omp for schedule (static,1) nowait
         for (int i = 0; i < n; i++){
            if (A->data[A->i[i]] != 0.0){
               u[i] = smooth_weight * f[i] / A->data[A->i[i]];
            }
         }
      }
      else{
	 #pragma omp for schedule (static,1) nowait
         for (int i = 0; i < n; i++) u_prev[i] = u[i];
	 #pragma omp for schedule (static,1) nowait
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
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
         #pragma omp for
         for (int i = 0; i < n; i++){
            u[i] = f[i] / all_data->matrix.L1_row_norm[level][i];
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

void SMEM_Async_Parfor_GaussSeidelT(AllData *all_data,
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
      for (int i = n-1; i >= 0; i--){
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
   int num_threads = omp_get_num_threads();
   int ns = all_data->thread.A_ns[level][t];
   int ne = all_data->thread.A_ne[level][t];

   
   //int size = n/num_threads;
   //int rest = n - size*num_threads;
   //if (t < rest)
   //{
   //   ns = t*size+t;
   //   ne = (t+1)*size+t+1;
   //}
   //else
   //{
   //   ns = t*size+rest;
   //   ne = (t+1)*size+rest;
   //}

   double smooth_weight = 1.0;//all_data->input.smooth_weight;

   double *diag_scale;
   if (all_data->input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL){
      hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
      diag_scale = hypre_ParAMGDataL1Norms(amg_data)[level];
      //diag_scale = all_data->matrix.L1_row_norm[level];
   }
   else {
      diag_scale = all_data->matrix.A_diag[level];
   }

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
         for (int i = ns; i < ne; i++) u[i] = 0.0;
         for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
                  ii = A->j[jj];
                  if (ii >= ns && ii < ne){
                     res -= A->data[jj] * u[ii];
                  }
               }
               u[i] = smooth_weight * res / diag_scale[i];
            }
         }
      }
      else{
         //#pragma omp for
         //for (int i = 0; i < n; i++) u_prev[i] = u[i]; 
         for (int i = ns; i < ne; i++) u_prev[i] = u[i];
         SMEM_Barrier(all_data, all_data->thread.global_barrier_flags);
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
               u[i] += smooth_weight * res / diag_scale[i];
            }
         }
      }
      SMEM_Barrier(all_data, all_data->thread.global_barrier_flags);
      //#pragma omp barrier
   }
}

void SMEM_Sync_Parfor_HybridJacobiGaussSeidelT(AllData *all_data,
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

   double smooth_weight = 1.0;//all_data->input.smooth_weight;

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
         for (int i = ne-1; i >= ns; i--) u[i] = 0.0;
         for (int i = ne-1; i >= ns; i--){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
                  ii = A->j[jj];
                  if (ii >= ns && ii < ne){
                     res -= A->data[jj] * u[ii];
                  }
               }
               u[i] = smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      else{
         for (int i = ne-1; i >= ns; i--) u_prev[i] = u[i];
         SMEM_Barrier(all_data, all_data->thread.global_barrier_flags);
	 for (int i = ne-1; i >= ns; i--){
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
               u[i] += smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      SMEM_Barrier(all_data, all_data->thread.global_barrier_flags);
     // #pragma omp barrier
   }
}

void SMEM_Sync_Jacobi(AllData *all_data,
                      hypre_CSRMatrix *A,
                      HYPRE_Real *f,
                      HYPRE_Real *u,
                      HYPRE_Real *u_prev,
                      int num_sweeps,
                      int level,
                      int ns, int ne)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);
   double smooth_weight = all_data->input.smooth_weight;

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
        // for (int j = ns; j < ne; j++){
	//    int i = all_data->vector.i[level][j];
	 for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               u[i] = smooth_weight * f[i] / A->data[A->i[i]];
            }
         }
      }
      else{
         for (int i = ns; i < ne; i++) u_prev[i] = u[i];
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
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
      //#pragma omp barrier
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
   }
}

void SMEM_Sync_L1Jacobi(AllData *all_data,
                        hypre_CSRMatrix *A,
                        HYPRE_Real *f,
                        HYPRE_Real *u,
                        HYPRE_Real *u_prev,
                        int num_sweeps,
                        int level,
                        int ns, int ne)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
         for (int i = ns; i < ne; i++){
            u[i] = f[i] / all_data->matrix.L1_row_norm[level][i];
         }
      }
      else{
         for (int i = ns; i < ne; i++) u_prev[i] = u[i];
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
         for (int i = ns; i < ne; i++){
            res = f[i];
            for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
               ii = A->j[jj];
               res -= A->data[jj] * u_prev[ii];
            }
            u[i] += res / all_data->matrix.L1_row_norm[level][i];
         }
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
   }
}

void SMEM_SemiAsync_GaussSeidel(AllData *all_data,
                                hypre_CSRMatrix *A,
                                HYPRE_Real *f,
                                HYPRE_Real *u,
                                int num_sweeps,
                                int level,
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
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
   }
}

void SMEM_Async_GaussSeidel(AllData *all_data,
                            hypre_CSRMatrix *A,
                            HYPRE_Real *f,
                            HYPRE_Real *u,
                            int num_sweeps,
                            int level,
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
   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
}

void SMEM_Async_GaussSeidelT(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *f,
                             HYPRE_Real *u,
                             int num_sweeps,
                             int level,
                             int ns, int ne)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   for (int k = 0; k < num_sweeps; k++){
      for (int i = ne-1; i >= ns; i--){
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
   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
}

void SMEM_Sync_HybridJacobiGaussSeidel(AllData *all_data,
                                       hypre_CSRMatrix *A,
                                       HYPRE_Real *f,
                                       HYPRE_Real *u,
                                       HYPRE_Real *u_prev,
                                       int num_sweeps,
                                       int level,
                                       int ns, int ne)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);
   double smooth_weight = 1.0;//all_data->input.smooth_weight;

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
         for (int i = ns; i < ne; i++) u[i] = 0.0;
         for (int i = ns; i < ne; i++){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
                  ii = A->j[jj];
                  if (ii >= ns && ii < ne){
                     res -= A->data[jj] * u[ii];
                  }
               }
               u[i] = smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      else{
         for (int i = ns; i < ne; i++) u_prev[i] = u[i];
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
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
               u[i] += smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      //#pragma omp barrier
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
   }
}

void SMEM_Sync_HybridJacobiGaussSeidelT(AllData *all_data,
                                        hypre_CSRMatrix *A,
                                        HYPRE_Real *f,
                                        HYPRE_Real *u,
                                        HYPRE_Real *u_prev,
                                        int num_sweeps,
                                        int level,
                                        int ns, int ne)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);
   double smooth_weight = 1.0;//all_data->input.smooth_weight;

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->grid.zero_flags[level] == 1){
         for (int i = ne-1; i >= ns; i--) u[i] = 0.0;
         for (int i = ne-1; i >= ns; i--){
            if (A->data[A->i[i]] != 0.0){
               res = f[i];
               for (int jj = A->i[i]; jj < A->i[i+1]; jj++){
                  ii = A->j[jj];
                  if (ii >= ns && ii < ne){
                     res -= A->data[jj] * u[ii];
                  }
               }
               u[i] = smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      else{
         for (int i = ne-1; i >= ns; i--) u_prev[i] = u[i];
         SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
	 for (int i = ne-1; i >= ns; i--){
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
               u[i] += smooth_weight * res / A->data[A->i[i]];
            }
         }
      }
      //#pragma omp barrier
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
   }
}

void SMEM_Sync_SymmetricJacobi(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *f,
                               HYPRE_Real *u,
                               HYPRE_Real *y,
                               HYPRE_Real *r,
                               int num_sweeps,
                               int level,
                               int ns, int ne)
{
   int k = 0;
   int tid = omp_get_thread_num();
   if (all_data->grid.zero_flags[level] == 1){
      for (int i = ns; i < ne; i++){
         r[i] = f[i];
      }
   }
   else {
      SMEM_Residual(all_data, A, f, u, y, r, ns, ne);
   }
   while(1){
      for (int i = ns; i < ne; i++){
         r[i] *= all_data->input.smooth_weight / A->data[A->i[i]];
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);

      int ms, me;
      if (level == 0 && all_data->input.res_compute_type == GLOBAL){
         ms = all_data->thread.A_ns_global[tid];
	 me = all_data->thread.A_ne_global[tid];
      }
      else {
         ms = ns; me = ne;
      }

      SMEM_MatVec(all_data, A, r, y, ms, me);
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);

      for (int i = ms; i < me; i++){
         r[i] = (2.0 * A->data[A->i[i]] * r[i] / all_data->input.smooth_weight) - y[i];
         r[i] *= all_data->input.smooth_weight / A->data[A->i[i]];
      }
      if (all_data->grid.zero_flags[level] == 1){
         for (int i = ms; i < me; i++){
            u[i] = r[i];
         }
      }
      else{
         for (int i = ms; i < me; i++){
            u[i] += r[i];
         }
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
      k++;
      if (k == num_sweeps){
         break;
      }
      SMEM_Residual(all_data, A, f, u, y, r, ms, me);
   }
}

void SMEM_Sync_SymmetricL1Jacobi(AllData *all_data,
                                 hypre_CSRMatrix *A,
                                 HYPRE_Real *f,
                                 HYPRE_Real *u,
                                 HYPRE_Real *y,
                                 HYPRE_Real *r,
                                 int num_sweeps,
                                 int level,
                                 int ns, int ne)
{
   int k = 0;
   int tid = omp_get_thread_num();
   if (all_data->grid.zero_flags[level] == 1){
      for (int i = ns; i < ne; i++){
         r[i] = f[i];
      }
   }
   else {
      SMEM_Residual(all_data, A, f, u, y, r, ns, ne);
   }
   while(1){
      for (int i = ns; i < ne; i++){
         r[i] /= all_data->matrix.L1_row_norm[level][i];
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);

      int ms, me;
      if (level == 0 && all_data->input.res_compute_type == GLOBAL){
         ms = all_data->thread.A_ns_global[tid];
         me = all_data->thread.A_ne_global[tid];
      }
      else {
         ms = ns; me = ne;
      }
      SMEM_MatVec(all_data, A, r, y, ms, me);
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);

      for (int i = ms; i < me; i++){
         r[i] = (2.0 * all_data->matrix.L1_row_norm[level][i] * r[i]) - y[i];
         r[i] /= all_data->matrix.L1_row_norm[level][i];
      }
      if (all_data->grid.zero_flags[level] == 1){
         for (int i = ns; i < me; i++){
            u[i] = r[i];
         }
      }
      else{
         for (int i = ns; i < me; i++){
            u[i] += r[i];
         }
      }
      SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
      k++;
      if (k == num_sweeps){
         break;
      }
      SMEM_Residual(all_data, A, f, u, y, r, ns, ne);
   }
}
