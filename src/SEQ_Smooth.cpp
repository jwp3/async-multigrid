#include "Main.hpp"


void SEQ_Jacobi(AllData *all_data,
                hypre_CSRMatrix *A,
                HYPRE_Real *f,
                HYPRE_Real *u,
                HYPRE_Real *u_prev,
                int num_sweeps)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   double smooth_weight = all_data->input.smooth_weight;

   for (int i = 0; i < n; i++) u_prev[i] = u[i];
   for (int k = 0; k < num_sweeps; k++){
      for (int i = 0; i < n; i++){
         if (A_data[A_i[i]] != 0.0)
         {
            res = f[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
               ii = A_j[jj];
               res -= A_data[jj] * u_prev[ii];
            }
            u[i] += smooth_weight * res / A_data[A_i[i]];
         }
      }
   }
}

void SEQ_GaussSeidel(AllData *all_data,
                     hypre_CSRMatrix *A,
                     HYPRE_Real *f,
                     HYPRE_Real *u,
                     int num_sweeps)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   for (int k = 0; k < num_sweeps; k++){
      for (int i = 0; i < n; i++){
         if (A_data[A_i[i]] != 0.0)
         {
            res = f[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
               ii = A_j[jj];
               res -= A_data[jj] * u[ii];
            }
            u[i] += res / A_data[A_i[i]];
         }
      }
   }
}

