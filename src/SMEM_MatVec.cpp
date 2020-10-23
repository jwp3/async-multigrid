#include "Main.hpp"
#include "Misc.hpp"

void SMEM_Sync_Parfor_MatVec(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *x,
                             HYPRE_Real *y)
{
   double Axi;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
  
   #pragma omp for
   for (int i = 0; i < num_rows; i++){
      Axi = 0.0;
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         Axi += A_data[jj] * x[A_j[jj]];
      }
      y[i] = Axi;
   }
}

void SMEM_Sync_Parfor_MatVecT(AllData *all_data,
                              hypre_CSRMatrix *A,
                              HYPRE_Real *x,
                              HYPRE_Real *y,
                              HYPRE_Real *y_expand)
{
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);

   int tid = omp_get_thread_num();
   int offset = num_cols * tid;

   #pragma omp for
   for (int i = 0; i < num_rows; i++){
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         y_expand[offset + A_j[jj]] += A_data[jj] * x[i];
      }
   }

   #pragma omp for
   for (int i = 0; i < num_cols; i++){
      y[i] = 0;
      for (int j = 0; j < all_data->input.num_threads; j++){
         int jj = j*num_cols + i;
         y[i] += y_expand[jj];
         y_expand[jj] = 0;
      }
   }
}

void SMEM_Sync_Parfor_Residual(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *b,
                               HYPRE_Real *x,
                               HYPRE_Real *y,
                               HYPRE_Real *r)
{
   double Axi;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);

   int chunk_size = 1;//num_rows/all_data->input.num_threads;

   #pragma omp for schedule(static, chunk_size)
   for (int i = 0; i < num_rows; i++){
      Axi = 0.0;
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         Axi += A_data[jj] * x[A_j[jj]];
      }
      r[i] = b[i] - Axi;
   }
}

//void SMEM_Sync_Parfor_Residual(AllData *all_data,
//                               hypre_CSRMatrix *A,
//                               HYPRE_Real *b,
//                               HYPRE_Real *x,
//                               HYPRE_Real *y,
//                               HYPRE_Real *r)
//{
//   HYPRE_Int n = hypre_CSRMatrixNumRows(A);
//
//   SMEM_Sync_Parfor_MatVec(all_data, A, x, y);
//   int tid = omp_get_thread_num();
//
//   #pragma omp for
//   for (int i = 0; i < n; i++)
//   {
//      r[i] = b[i] - y[i];
//   }
//}

void SMEM_Sync_Parfor_SpGEMV(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *x,
                             HYPRE_Real *b,
                             HYPRE_Real alpha, 
                             HYPRE_Real beta,
                             HYPRE_Real *y)
{
   double Axi;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);

   #pragma omp for
   for (int i = 0; i < num_rows; i++){
      Axi = 0.0;
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         Axi += A_data[jj] * x[A_j[jj]];
      }
      y[i] = beta*b[i] + alpha*Axi;
   }
}

void SMEM_Async_Parfor_MatVec(AllData *all_data,
                             hypre_CSRMatrix *A,
                             HYPRE_Real *x,
                             HYPRE_Real *y)
{
   double Axi;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);

   #pragma omp for schedule (static,1) nowait
   for (int i = 0; i < num_rows; i++){
      Axi = 0.0;
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++)
      {
         Axi += A_data[jj] * x[A_j[jj]];
      }
      y[i] = Axi;
   }
}

void SMEM_Async_Parfor_Residual(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *b,
                               HYPRE_Real *x,
                               HYPRE_Real *y,
                               HYPRE_Real *r)
{
   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   SMEM_Async_Parfor_MatVec(all_data, A, x, y);

   #pragma omp for schedule (static,1) nowait
   for (int i = 0; i < n; i++)
   {
      r[i] = b[i] - y[i];
   }
}

void SMEM_MatVec(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *x,
                 HYPRE_Real *y,
                 int ns, int ne)
{
   double Axi;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);

   for (int i = ns; i < ne; i++){
      Axi = 0.0;
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++)
      {
         Axi += A_data[jj] * x[A_j[jj]];
      }
      y[i] = Axi;
   }
}

void SMEM_Residual(AllData *all_data,
                   hypre_CSRMatrix *A,
                   HYPRE_Real *b,
                   HYPRE_Real *x,
                   HYPRE_Real *y,
                   HYPRE_Real *r,
                   int ns, int ne)
{
   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   SMEM_MatVec(all_data, A, x, y, ns, ne);

   for (int i = ns; i < ne; i++){
      double ri = b[i] - y[i];
      r[i] = ri;
   }
}

void SMEM_JacobiIterMat_MatVec(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *y,
                               HYPRE_Real *r,
                               int ns, int ne,
                               int thread_level)
{
   SMEM_MatVec(all_data, A, r, y, ns, ne);
   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);

   for (int i = ns; i < ne; i++)
   {
      if (A->data[A->i[i]] != 0.0)
      {
         r[i] -= all_data->input.smooth_weight * y[i] / A->data[A->i[i]];
      }
   }
}

//void SMEM_JacobiSymmIterMat_MatVec(AllData *all_data,
//                                   hypre_CSRMatrix *A,
//                                   HYPRE_Real *y,
//                                   HYPRE_Real *r,
//                                   int ns, int ne,
//                                   int thread_level)
//{
//   for (int i = ns; i < ne; i++)
//   {
//      if (A->data[A->i[i]] != 0.0)
//      {
//         r[i] *= all_data->input.smooth_weight / A->data[A->i[i]];
//      }
//   }
//   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
//
//   SMEM_MatVec(all_data, A, r, y, ns, ne);
//   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, thread_level);
//
//   for (int i = ns; i < ne; i++)
//   {
//      if (A->data[A->i[i]] != 0.0)
//      {
//         r[i] = (2.0 * A->data[A->i[i]] * r[i] / all_data->input.smooth_weight) - 
//                (all_data->input.smooth_weight * y[i]);
//         r[i] *= all_data->input.smooth_weight / A->data[A->i[i]];
//      }
//   }
//}


void SMEM_MatVec2(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *x,
                 HYPRE_Real *y,
                 int ns, int ne,
                 double alpha, double beta)
{
   double Axi;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);

   for (int i = ns; i < ne; i++){
      Axi = 0.0;
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++)
      {
         Axi += A_data[jj] * x[A_j[jj]];
      }
      y[i] = beta * x[i] + alpha * Axi;
   }
}
