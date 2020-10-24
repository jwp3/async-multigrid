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
  // double Axi;
  // HYPRE_Int *A_i = hypre_CSRMatrixI(A);
  // HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
  // HYPRE_Real *A_data = hypre_CSRMatrixData(A);
  // HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);

  // #pragma omp for
  // for (int i = 0; i < num_rows; i++){
  //    Axi = 0.0;
  //    for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
  //       Axi += A_data[jj] * x[A_j[jj]];
  //    }
  //    y[i] = beta*b[i] + alpha*Axi;
  // }

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_rownnz = hypre_CSRMatrixNumRownnz(A);
   HYPRE_Real tempx;
   HYPRE_Int m;

   HYPRE_Real xpar = 0.7;
   HYPRE_Real temp = beta / alpha;


  // if (num_rownnz < xpar * num_rows) {
  //     HYPRE_Int *A_rownnz = hypre_CSRMatrixRownnz(A);
  //    /*-----------------------------------------------------------------------
  //     * y = (beta/alpha)*y
  //     *-----------------------------------------------------------------------*/

  //    if (temp != 1.0){
  //       if (temp == 0.0){
  //          #pragma omp for HYPRE_SMP_SCHEDULE
  //          for (int i = 0; i < num_rows; i++){
  //             y[i] = 0.0;
  //          }
  //       }
  //       else {
  //          #pragma omp for HYPRE_SMP_SCHEDULE
  //          for (int i = 0; i < num_rows; i++){
  //             y[i] = b[i] * temp;
  //          }
  //       }
  //    }
  //    else {
  //       #pragma omp for HYPRE_SMP_SCHEDULE
  //       for (int i = 0; i < num_rows; i++){
  //          y[i] = b[i];
  //       }
  //    }


  //    /*-----------------------------------------------------------------
  //     * y += A*x
  //     *-----------------------------------------------------------------*/

  //    #pragma omp for HYPRE_SMP_SCHEDULE
  //    for (int i = 0; i < num_rownnz; i++){
  //       m = A_rownnz[i];
  //       tempx = 0;
  //       for (int jj = A_i[m]; jj < A_i[m+1]; jj++){
  //          tempx += A_data[jj] * x[A_j[jj]];
  //       }
  //       y[m] += tempx;
  //    }

  //    /*-----------------------------------------------------------------
  //     * y = alpha*y
  //     *-----------------------------------------------------------------*/

  //    if (alpha != 1.0){
  //       #pragma omp for HYPRE_SMP_SCHEDULE
  //       for (int i = 0; i < num_rows; i++){
  //          y[i] *= alpha;
  //       }
  //    }
  // }
  // else {
      HYPRE_Int iBegin = hypre_CSRMatrixGetLoadBalancedPartitionBegin(A);
      HYPRE_Int iEnd = hypre_CSRMatrixGetLoadBalancedPartitionEnd(A);

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
   //}
   #pragma omp barrier
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
