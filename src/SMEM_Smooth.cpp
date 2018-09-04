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

void SMEM_Sync_Parfor_IBlockJacobi_GaussSeidel(AllData *all_data,
                                               hypre_CSRMatrix *A,
                                               HYPRE_Real *f,
                                               HYPRE_Real *u,
                                               int num_sweeps)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   for (int k = 0; k < num_sweeps; k++){
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

