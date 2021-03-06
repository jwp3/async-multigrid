#include "Main.hpp"
#include "Misc.hpp"
#include "SMEM_MatVec.hpp"

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
   SMEM_Sync_Parfor_SpGEMV(all_data, A, x, b, -1.0, 1.0, r);
}

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

void SMEM_Sync_Residual(AllData *all_data,
                        hypre_CSRMatrix *A,
                        HYPRE_Real *b,
                        HYPRE_Real *x,
                        HYPRE_Real *y,
                        HYPRE_Real *r)
{
   SMEM_Sync_SpGEMV(all_data, A, x, b, -1.0, 1.0, r);
}


void SMEM_Sync_SpGEMV(AllData *all_data,
                      hypre_CSRMatrix *A,
                      HYPRE_Real *x,
                      HYPRE_Real *b,
                      HYPRE_Real alpha,
                      HYPRE_Real beta,
                      HYPRE_Real *y)
{
   HYPRE_Int iBegin = hypre_CSRMatrixGetLoadBalancedPartitionBegin(A);
   HYPRE_Int iEnd = hypre_CSRMatrixGetLoadBalancedPartitionEnd(A);
   
   SMEM_SpGEMV(all_data, A, x, b, alpha, beta, y, iBegin, iEnd);   

   #pragma omp barrier
}


void SMEM_SpGEMV(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *x,
                 HYPRE_Real *b,
                 HYPRE_Real alpha, 
                 HYPRE_Real beta,
                 HYPRE_Real *y,
                 int iBegin, int iEnd)
{
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_rownnz = hypre_CSRMatrixNumRownnz(A);
   HYPRE_Real tempx;
   HYPRE_Int m;

   HYPRE_Real xpar = 0.7;
   HYPRE_Real temp = beta / alpha;

   if (temp == 0){
      if (alpha == 1){
         for (int i = iBegin; i < iEnd; i++){
            tempx = 0.0;
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx += A_data[jj] * x[A_j[jj]];
            }
            y[i] = tempx;
         }
      } // y = A*x
      else if (alpha == -1){
         for (int i = iBegin; i < iEnd; i++){
            tempx = 0.0;
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx -= A_data[jj] * x[A_j[jj]];
            }
            y[i] = tempx;
         }
      } // y = -A*x
      else {
         for (int i = iBegin; i < iEnd; i++){
            tempx = 0.0;
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx += A_data[jj] * x[A_j[jj]];
            }
            y[i] = alpha*tempx;
         }
      } // y = alpha*A*x
   } // temp == 0
   else if (temp == -1){ // beta == -alpha
      if (alpha == 1){
         for (int i = iBegin; i < iEnd; i++){
            tempx = -b[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx += A_data[jj] * x[A_j[jj]];
            }
            y[i] = tempx;
         }
      } // y = A*x - y
      else if (alpha == -1) {
         for (int i = iBegin; i < iEnd; i++){
            tempx = b[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx -= A_data[jj] * x[A_j[jj]];
            }
            y[i] = tempx;
         }
      } // y = -A*x + y
      else {
         for (int i = iBegin; i < iEnd; i++){
            tempx = -b[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx += A_data[jj] * x[A_j[jj]];
            }
            y[i] = alpha*tempx;
         }
      } // y = alpha*(A*x - y)
   } // temp == -1
   else if (temp == 1){
      if (alpha == 1){
         for (int i = iBegin; i < iEnd; i++){
            tempx = b[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx += A_data[jj] * x[A_j[jj]];
            }
            y[i] = tempx;
         }
      } // y = A*x + y
      else if (alpha == -1){
         for (int i = iBegin; i < iEnd; i++){
            tempx = -b[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx -= A_data[jj] * x[A_j[jj]];
            }
            y[i] = tempx;
         }
      } // y = -A*x - y
      else {
         for (int i = iBegin; i < iEnd; i++){
            tempx = b[i];
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx += A_data[jj] * x[A_j[jj]];
            }
            y[i] = alpha*tempx;
         }
      } // y = alpha*(A*x + y)
   }
   else {
      if (alpha == 1) {
         for (int i = iBegin; i < iEnd; i++){
            tempx = b[i]*temp;
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx += A_data[jj] * x[A_j[jj]];
            }
            y[i] = tempx;
         }
      } // y = A*x + temp*y
      else if (alpha == -1){
         for (int i = iBegin; i < iEnd; i++){
            tempx = -b[i]*temp;
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx -= A_data[jj] * x[A_j[jj]];
            }
            y[i] = tempx;
         }
      } // y = -A*x - temp*y
      else{
         for (int i = iBegin; i < iEnd; i++){
            tempx = b[i]*temp;
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               tempx += A_data[jj] * x[A_j[jj]];
            }
            y[i] = alpha*tempx;
         }
      } // y = alpha*(A*x + temp*y)
   } // temp != 0 && temp != -1 && temp != 1
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
   int tid = omp_get_thread_num();

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);

   for (int i = ns; i < ne; i++){
      Axi = 0.0;
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         Axi += A_data[jj] * x[A_j[jj]];
      }
      y[i] = Axi;
   }
}

void SMEM_MatVecT(AllData *all_data,
                  hypre_CSRMatrix *A,
                  HYPRE_Real *x,
                  HYPRE_Real *y,
                  HYPRE_Real *y_expand,
                  int ns_row, int ne_row,
                  int ns_col, int ne_col,
                  int t, int level)
{
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);

   int offset = num_cols * t;

   for (int i = ns_row; i < ne_row; i++){
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         y_expand[offset + A_j[jj]] += A_data[jj] * x[i];
      }
   }

   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);

   for (int i = ns_col; i < ne_col; i++){
      y[i] = 0;
      for (int j = 0; j < all_data->thread.level_threads[level].size(); j++){
         int jj = j*num_cols + i;
         y[i] += y_expand[jj];
         y_expand[jj] = 0;
      }
   }

   SMEM_LevelBarrier(all_data, all_data->thread.barrier_flags, level);
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

void SMEM_Sync_Parfor_Restrict(AllData *all_data,
                               hypre_CSRMatrix *R,
                               HYPRE_Real *v_fine,
                               HYPRE_Real *v_coarse,
                               int fine_grid, int coarse_grid)
{
   if (all_data->input.construct_R_flag == 1){
      SMEM_Sync_Parfor_MatVec(all_data, R, v_fine, v_coarse);
   }
   else {
      SMEM_Sync_Parfor_MatVecT(all_data, R, v_fine, v_coarse, all_data->vector.y_expand[fine_grid]);
   }
}

void SMEM_Restrict(AllData *all_data,
                   hypre_CSRMatrix *R,
                   HYPRE_Real *v_fine,
                   HYPRE_Real *v_coarse,
                   int fine_grid, int coarse_grid,
                   int ns_row, int ne_row, int ns_col, int ne_col,
                   int t, int level)
{
   if (all_data->input.construct_R_flag == 1){
      SMEM_MatVec(all_data, R, v_fine, v_coarse, ns_row, ne_row);
   }
   else {
      SMEM_MatVecT(all_data, R, v_fine, v_coarse, all_data->level_vector[level].y_expand[fine_grid], ns_row, ne_row, ns_col, ne_col, t, level);
   }
}
