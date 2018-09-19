#include "Main.hpp"

void SEQ_MatVec(AllData *all_data,
                hypre_CSRMatrix *A,
                HYPRE_Real *x,
                HYPRE_Real *y)
{
   double Axi;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);

   
   for (int i = 0; i < num_rows; i++){
      Axi = 0.0;
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         Axi += A_data[jj] * x[A_j[jj]];
      }
      y[i] = Axi;
   }
}

void SEQ_MatVecT(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *x,
                 HYPRE_Real *y)
{
   int j;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);

   for (int i = 0; i < num_cols; i++) y[i] = 0; 
   for (int i = 0; i < num_rows; i++){
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         j = A_j[jj];
         y[j] += A_data[jj] * x[i];
      }
   }
}

void SEQ_Residual(AllData *all_data,
                  hypre_CSRMatrix *A,
                  HYPRE_Real *b,
                  HYPRE_Real *x,
                  HYPRE_Real *y,
                  HYPRE_Real *r)
{
   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   SEQ_MatVec(all_data, A, x, y);

   for (int i = 0; i < n; i++)
   {
      r[i] = b[i] - y[i];
   }
}
