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
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++)
      {
         Axi += A_data[jj] * x[A_j[jj]];
      }
      y[i] = Axi;
   }
}

void SMEM_Sync_Parfor_Residual(AllData *all_data,
                               hypre_CSRMatrix *A,
                               HYPRE_Real *b,
                               HYPRE_Real *x,
                               HYPRE_Real *y,
                               HYPRE_Real *r)
{
   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   SMEM_Sync_Parfor_MatVec(all_data, A, x, y);

   #pragma omp for
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
