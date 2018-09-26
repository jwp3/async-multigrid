#include "Main.hpp"
#include "SEQ_MatVec.hpp"

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

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->vector.zero_flag == 1){
         for (int i = 0; i < n; i++){
            if (A_data[A_i[i]] != 0.0){
               res = f[i];
               u[i] += smooth_weight * res / A_data[A_i[i]];
            }
         }
      }
      else{
         for (int i = 0; i < n; i++) u_prev[i] = u[i];
         for (int i = 0; i < n; i++){
            if (A_data[A_i[i]] != 0.0){
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
}

void SEQ_L1Jacobi(AllData *all_data,
                  hypre_CSRMatrix *A,
                  HYPRE_Real *f,
                  HYPRE_Real *u,
                  HYPRE_Real *u_prev,
                  double *L1_row_norm,
                  int num_sweeps)
{
   HYPRE_Int ii;
   HYPRE_Real res;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int n = hypre_CSRMatrixNumRows(A);

   for (int k = 0; k < num_sweeps; k++){
      if (k == 0 && all_data->vector.zero_flag == 1){
         for (int i = 0; i < n; i++){
            if (A_data[A_i[i]] != 0.0){
               res = f[i];
               u[i] += res / L1_row_norm[i];
            }
         }
      }
      else{
         for (int i = 0; i < n; i++) u_prev[i] = u[i];
         for (int i = 0; i < n; i++){
            res = f[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
               ii = A_j[jj];
               res -= A_data[jj] * u_prev[ii];
            }
            u[i] += res / L1_row_norm[i];
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

void SEQ_SymmetricJacobi(AllData *all_data,
                         hypre_CSRMatrix *A,
                         HYPRE_Real *f,
                         HYPRE_Real *u,
                         HYPRE_Real *y,
                         HYPRE_Real *r,
                         int num_sweeps,
                         int level)
{
   int n = hypre_CSRMatrixNumRows(A);
   int k = 0;
   for (int i = 0; i < n; i++){
      r[i] = f[i];
   }
   while(1){
      for (int i = 0; i < n; i++){
         if (A->data[A->i[i]] != 0.0){
            r[i] *= all_data->input.smooth_weight / A->data[A->i[i]];
         }
      }

      SEQ_MatVec(all_data, A, r, y);

      for (int i = 0; i < n; i++){
         if (A->data[A->i[i]] != 0.0){
            r[i] = (2.0 * A->data[A->i[i]] * r[i] / all_data->input.smooth_weight) - y[i];
            r[i] *= all_data->input.smooth_weight / A->data[A->i[i]];
         }
         u[i] += r[i];
      }
      k++;
      if (k == num_sweeps){
         break;
      }
      SEQ_Residual(all_data, A, f, u, y, r);
   }
}

void SEQ_SymmetricL1Jacobi(AllData *all_data,
                           hypre_CSRMatrix *A,
                           HYPRE_Real *f,
                           HYPRE_Real *u,
                           HYPRE_Real *y,
                           HYPRE_Real *r,
                           int num_sweeps,
                           int level)
{
   int n = hypre_CSRMatrixNumRows(A);
   int k = 0;
   for (int i = 0; i < n; i++){
      r[i] = f[i];
   }
   while(1){
      for (int i = 0; i < n; i++){
         r[i] /= all_data->matrix.L1_row_norm[level][i];
      }

      SEQ_MatVec(all_data, A, r, y);

      for (int i = 0; i < n; i++){
         r[i] = (2.0 * all_data->matrix.L1_row_norm[level][i] * r[i]) - y[i];
         r[i] /= all_data->matrix.L1_row_norm[level][i];
         u[i] += r[i];
      }
      k++;
      if (k == num_sweeps){
         break;
      }
      SEQ_Residual(all_data, A, f, u, y, r);
   }
}
