#include "Main.hpp"
#include "SEQ_Setup.hpp"
#include "Misc.hpp"

void HypreParToSeq(void *amg_vdata,
                   AllData *all_data);

void StdVector_to_CSR(hypre_CSRMatrix *A,
                      std::vector<std::vector<HYPRE_Int>> j_vector,
                      std::vector<std::vector<HYPRE_Real>> data_vector);

void CSR_Transpose(hypre_CSRMatrix *A,
                   hypre_CSRMatrix *AT);

void SEQ_Setup(void *amg_vdata,
               AllData *all_data)
{
   HypreParToSeq(amg_vdata, all_data);  
}

void HypreParToSeq(void *amg_vdata,
                   AllData *all_data)
{
   HYPRE_Int n, nnz;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*)amg_vdata;
  // printf("%d\n", hypre_ParAMGDataNumLevels(amg_data));
   
   hypre_ParCSRMatrix **parA;
   hypre_ParCSRMatrix **parP;
   hypre_ParCSRMatrix **parR;

   parA = hypre_ParAMGDataAArray(amg_data);
   parP = hypre_ParAMGDataPArray(amg_data);
   parR = hypre_ParAMGDataRArray(amg_data);

   all_data->grid.num_levels = (int)hypre_ParAMGDataNumLevels(amg_data);
   all_data->grid.n = (int *)malloc(all_data->grid.num_levels * sizeof(int));

   all_data->matrix.A =
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix));
   all_data->matrix.P =
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix));
   all_data->matrix.R =
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix));

   all_data->vector.f =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.u =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.u_prev =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.u_coarse =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.u_coarse_prev =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.u_fine =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.u_fine_prev =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.y =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.r =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.r_coarse =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.r_fine =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));
   all_data->vector.e =
      (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real));

   for (int level = 0; level < all_data->grid.num_levels; level++){

      all_data->matrix.A[level] = hypre_ParCSRMatrixDiag(parA[level]);      
      if (level < all_data->grid.num_levels-1){
         all_data->matrix.P[level] = hypre_ParCSRMatrixDiag(parP[level]);
         all_data->matrix.R[level] = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
         CSR_Transpose(hypre_ParCSRMatrixDiag(parR[level]), all_data->matrix.R[level]);
        // all_data->matrix.R[level] = hypre_ParCSRMatrixDiag(parR[level]);
      }

      n = hypre_CSRMatrixNumRows(all_data->matrix.A[level]);
      nnz = hypre_CSRMatrixNumNonzeros(all_data->matrix.A[level]);

      all_data->vector.f[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.u[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.u_prev[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.u_coarse[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.u_coarse_prev[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.u_fine[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.u_fine_prev[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.y[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.r[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.r_coarse[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.r_fine[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.e[level] =
         (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));

      all_data->grid.n[level] = n;

      if (level == 0){
         for (int i = 0; i < n; i++) all_data->vector.f[level][i] = 1;
      }
      if (level == all_data->grid.num_levels-1){
         
         HYPRE_Int *A_i = hypre_CSRMatrixI(all_data->matrix.A[level]);
         HYPRE_Int *A_j = hypre_CSRMatrixJ(all_data->matrix.A[level]);
         HYPRE_Real *A_data = hypre_CSRMatrixData(all_data->matrix.A[level]);

         all_data->pardiso.csr.n = n;
         all_data->pardiso.csr.nnz = nnz;

         all_data->pardiso.csr.ja = (MKL_INT *)calloc(all_data->pardiso.csr.nnz, sizeof(MKL_INT));
         all_data->pardiso.csr.a = (double *)calloc(all_data->pardiso.csr.nnz, sizeof(double));
         all_data->pardiso.csr.ia = (MKL_INT *)calloc(all_data->pardiso.csr.n+1, sizeof(MKL_INT));

         for (int i = 0; i < all_data->pardiso.csr.nnz; i++){
            all_data->pardiso.csr.ja[i] = A_j[i];
            all_data->pardiso.csr.a[i] = A_data[i];
         }
         for (int i = 0; i < all_data->pardiso.csr.n+1; i++){
            all_data->pardiso.csr.ia[i] = A_i[i];
         }

         for (int i = 0; i < n; i++){
            QuicksortPair_int_dbl(all_data->pardiso.csr.ja,
                                  all_data->pardiso.csr.a,
                                  all_data->pardiso.csr.ia[i],
                                  all_data->pardiso.csr.ia[i+1]-1);
         }

         for (int i = 0; i < 64; i++){
            all_data->pardiso.info.iparm[i] = 0;
            all_data->pardiso.info.pt[i] = 0;
         }

         all_data->pardiso.info.mtype = 11;
         all_data->pardiso.info.nrhs = 1;
         all_data->pardiso.info.iparm[17] = -1;
         all_data->pardiso.info.iparm[18] = -1;
         all_data->pardiso.info.iparm[0] = 1;         /* No solver default */
         all_data->pardiso.info.iparm[1] = 0;         /* Fill-in reordering from METIS */
         all_data->pardiso.info.iparm[7] = 1;
         all_data->pardiso.info.iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
         all_data->pardiso.info.iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
         all_data->pardiso.info.iparm[12] = 1;
         all_data->pardiso.info.iparm[24] = 1;
         all_data->pardiso.info.iparm[26] = 1;
         all_data->pardiso.info.iparm[34] = 1;        /* turn off 1-based indexing */
         all_data->pardiso.info.maxfct = 1;           /* Maximum number of numerical factorizations. */
         all_data->pardiso.info.mnum = 1;         /* Which factorization to use. */
         all_data->pardiso.info.msglvl = 0;           /* Print statistical information in file */
         all_data->pardiso.info.error = 0;            /* Initialize error flag */
        /* reordering and Symbolic factorization. this step also allocates
         *all memory that is necessary for the factorization. */
         all_data->pardiso.info.phase = 12; 
         PARDISO(all_data->pardiso.info.pt,
                 &(all_data->pardiso.info.maxfct),
                 &(all_data->pardiso.info.mnum),
                 &(all_data->pardiso.info.mtype),
                 &(all_data->pardiso.info.phase),
                 &(all_data->pardiso.csr.n),
                 all_data->pardiso.csr.a,
                 all_data->pardiso.csr.ia,
                 all_data->pardiso.csr.ja,
                 &(all_data->pardiso.info.idum),
                 &(all_data->pardiso.info.nrhs),
                 all_data->pardiso.info.iparm,
                 &(all_data->pardiso.info.msglvl),
                 &(all_data->pardiso.info.ddum),
                 &(all_data->pardiso.info.ddum),
                 &(all_data->pardiso.info.error));
         all_data->pardiso.info.phase = 33; 
      }
   }
}

void CSR_Transpose(hypre_CSRMatrix *A,
                   hypre_CSRMatrix *AT)
{
   int j;
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);

   HYPRE_Int A_num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int A_num_cols = hypre_CSRMatrixNumCols(A);
  
   hypre_CSRMatrixNumRows(AT) = A_num_cols;
   hypre_CSRMatrixNumCols(AT) = A_num_rows;
   hypre_CSRMatrixNumNonzeros(AT) = hypre_CSRMatrixNumNonzeros(A);

   std::vector<std::vector<HYPRE_Int>> 
      j_vector(A_num_cols, std::vector<HYPRE_Int>(0));
   std::vector<std::vector<HYPRE_Real>> 
      data_vector(A_num_cols, std::vector<HYPRE_Real>(0));

   
   
   for (int i = 0; i < A_num_rows; i++){
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         j = A_j[jj];
         j_vector[j].push_back(i);
         data_vector[j].push_back(A_data[jj]);
      }
   }
   
   StdVector_to_CSR(AT, j_vector, data_vector); 
}

void StdVector_to_CSR(hypre_CSRMatrix *A,
                      std::vector<std::vector<HYPRE_Int>> j_vector,
                      std::vector<std::vector<HYPRE_Real>> data_vector)
{
   int k;

   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(A);

   HYPRE_Int *A_i = (HYPRE_Int *)calloc(num_rows+1, sizeof(HYPRE_Int));
   HYPRE_Int *A_j = (HYPRE_Int *)calloc(nnz, sizeof(HYPRE_Int));
   HYPRE_Real *A_data = (HYPRE_Real *)calloc(nnz, sizeof(HYPRE_Real));


   A_i[0] = 0;
   for (int i = 0; i < num_rows; i++){
      A_i[i+1] = A_i[i] + j_vector[i].size();
   }

   k = 0;
   for (int i = 0; i < num_rows; i++){
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         A_j[k] = j_vector[i].back();
         A_data[k] = data_vector[i].back();
         j_vector[i].pop_back();
         data_vector[i].pop_back();
         k++;
      }
   }

   hypre_CSRMatrixI(A) = A_i;
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_data;
}
