#include "Main.hpp"
#include "SMEM_Setup.hpp"
#include "Misc.hpp"

void InitAlgebra(void *amg_vdata,
                 AllData *all_data);

void PartitionLevels(AllData *all_data);

void PartitionGrids(AllData *all_data);

void StdVector_to_CSR(hypre_CSRMatrix *A,
                      std::vector<std::vector<HYPRE_Int>> j_vector,
                      std::vector<std::vector<HYPRE_Real>> data_vector);

void CSR_Transpose(hypre_CSRMatrix *A,
                   hypre_CSRMatrix *AT);

void SetupThreads(AllData *all_data);

void SMEM_Setup(void *amg_vdata,
                AllData *all_data)
{
   InitAlgebra(amg_vdata, all_data);
   PartitionLevels(all_data);
   PartitionGrids(all_data);
}

void InitAlgebra(void *amg_vdata,
                 AllData *all_data)
{
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
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix *));
   all_data->matrix.P =
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix *));
   all_data->matrix.R =
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix *));

   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->matrix.A[level] = hypre_ParCSRMatrixDiag(parA[level]);
      all_data->grid.n[level] = hypre_CSRMatrixNumRows(all_data->matrix.A[level]);
      if (level < all_data->grid.num_levels-1){
         all_data->matrix.P[level] = hypre_ParCSRMatrixDiag(parP[level]);
         all_data->matrix.R[level] = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
         CSR_Transpose(hypre_ParCSRMatrixDiag(parR[level]), all_data->matrix.R[level]);
        // all_data->matrix.R[level] = hypre_ParCSRMatrixDiag(parR[level]);
      }
   }
   all_data->vector.u =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
   all_data->vector.y =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
   all_data->vector.r =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
   all_data->vector.f =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));

   all_data->grid.num_correct = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   all_data->grid.level_res_comp_count = (int *)calloc(all_data->grid.num_levels, sizeof(int));

   if (all_data->input.thread_part_type == ALL_LEVELS){

      all_data->level_vector =
         (VectorData *)malloc(all_data->grid.num_levels * sizeof(VectorData));

      for (int level = 0; level < all_data->grid.num_levels; level++){

         all_data->level_vector[level].f = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].u = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].u_prev = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].u_coarse =(HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].u_coarse_prev = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].u_fine = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].u_fine_prev = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].y = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].r = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].r_coarse = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].r_fine = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         all_data->level_vector[level].e = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      }

      for (int level = 0; level < all_data->grid.num_levels; level++){
         for (int inner_level = 0; inner_level < level+2; inner_level++){
            if (inner_level < all_data->grid.num_levels){
               int n = all_data->grid.n[inner_level];
               all_data->level_vector[level].f[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].u[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].u_prev[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].u_coarse[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].u_coarse_prev[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].u_fine[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].u_fine_prev[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].y[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].r[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].r_coarse[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].r_fine[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->level_vector[level].e[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
            }
         }
         if (level == 0){
            int n = all_data->grid.n[level];
            all_data->vector.f[level] =
               (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
            all_data->vector.r[level] =
               (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
            all_data->vector.u[level] =
               (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
            all_data->vector.y[level] =
               (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
         }
      }
   }
   else {
      all_data->vector.u_prev =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.u_coarse =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.u_coarse_prev =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.u_fine =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.u_fine_prev =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.r_coarse =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.r_fine =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.e =
         (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      for (int level = 0; level < all_data->grid.num_levels; level++){
         int n = all_data->grid.n[level];
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
      }
   }
   

   int level = all_data->grid.num_levels-1;         
   HYPRE_Int *A_i = hypre_CSRMatrixI(all_data->matrix.A[level]);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(all_data->matrix.A[level]);
   HYPRE_Real *A_data = hypre_CSRMatrixData(all_data->matrix.A[level]);

   all_data->pardiso.csr.n = all_data->grid.n[level];
   all_data->pardiso.csr.nnz = hypre_CSRMatrixNumNonzeros(all_data->matrix.A[level]);

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

   for (int i = 0; i < all_data->pardiso.csr.n; i++){
      BubblesortPair_int_double(&all_data->pardiso.csr.ja[all_data->pardiso.csr.ia[i]],
                                &all_data->pardiso.csr.a[all_data->pardiso.csr.ia[i]],
                                all_data->pardiso.csr.ia[i+1] - all_data->pardiso.csr.ia[i]);
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

void PartitionLevels(AllData *all_data)
{
   int num_threads = all_data->input.num_threads;
   int num_levels = all_data->grid.num_levels;
   int half_threads, n_threads;
   int num_level_threads;

   all_data->thread.thread_levels.resize(num_threads, std::vector<int>(0));
   all_data->thread.level_threads.resize(num_levels, std::vector<int>(0));
   all_data->thread.barrier_flags = (int **)malloc(num_levels * sizeof(int *));
   all_data->thread.barrier_root = (int *)malloc(num_levels * sizeof(int));

  
   if (all_data->input.thread_part_type == ALL_LEVELS){
      int tid = 0;
      for (int level = 0; level < num_levels; level++){
         all_data->thread.barrier_flags[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
      }
      if (all_data->input.thread_part_distr_type == HALF_THREADS){
         for (int level = 0; level < num_levels; level++){
           // printf("level %d:\n\t", level);
            if (num_threads == 1){
              // printf("%d ", tid);
               all_data->thread.thread_levels[tid].push_back(level);
               all_data->thread.level_threads[level].push_back(tid);
               all_data->thread.barrier_root[level] = tid;
               all_data->thread.barrier_flags[level][tid] = 0;
            }
            else{
               if (level == num_levels-1){
                  for (int t = tid; t < tid + num_threads; t++){
                    // printf("%d ", t);
                     all_data->thread.thread_levels[t].push_back(level);
                     all_data->thread.level_threads[level].push_back(t);
                  }
                  for (int t = tid; t < tid + half_threads; t++){
                     all_data->thread.barrier_flags[level][t] = 0;
                  }
                  tid += num_threads;
                  all_data->thread.barrier_root[level] = tid-1;
               }
               else {
                  half_threads = (int)ceil(((double)num_threads)/2.0);
                  for (int t = tid; t < tid + half_threads; t++){
                    // printf("%d ", t);
                     all_data->thread.thread_levels[t].push_back(level);
                     all_data->thread.level_threads[level].push_back(t);
                  }
                  for (int t = tid; t < tid + half_threads; t++){
                     all_data->thread.barrier_flags[level][t] = 0;
                  }
                  num_threads -= half_threads;
                  tid += half_threads;
                  all_data->thread.barrier_root[level] = tid-1;
               }
            }
           // printf("\n");
         }
      }
      else {
         int equal_threads = std::max(all_data->input.num_threads/all_data->grid.num_levels, 1);
         for (int level = 0; level < num_levels; level++){
           // printf("level %d:\n\t", level);
            if (num_threads == 1){
              // printf("%d ", tid);
               all_data->thread.thread_levels[tid].push_back(level);
               all_data->thread.level_threads[level].push_back(tid);
               all_data->thread.barrier_root[level] = tid;
               all_data->thread.barrier_flags[level][tid] = 0;
            }
            else{
               if (level == num_levels-1){
                  for (int t = tid; t < tid + num_threads; t++){
                    // printf("%d ", t);
                     all_data->thread.thread_levels[t].push_back(level);
                     all_data->thread.level_threads[level].push_back(t);
                  }
                  for (int t = tid; t < tid + half_threads; t++){
                     all_data->thread.barrier_flags[level][t] = 0;
                  }
                  tid += num_threads;
                  all_data->thread.barrier_root[level] = tid-1;
               }
               else {
                  for (int t = tid; t < tid + equal_threads; t++){
                    // printf("%d ", t);
                     all_data->thread.thread_levels[t].push_back(level);
                     all_data->thread.level_threads[level].push_back(t);
                  }
                  for (int t = tid; t < tid + equal_threads; t++){
                     all_data->thread.barrier_flags[level][t] = 0;
                  }
                  num_threads -= equal_threads;
                  tid += equal_threads;
                  all_data->thread.barrier_root[level] = tid-1;
               }
            }
           // printf("\n");
         }
      }
   }
   else{
      for (int level = 0; level < num_levels; level++){
         all_data->thread.barrier_flags[level] = (int *)malloc(num_threads * sizeof(int));
         for (int t = 0; t < num_threads; t++){
            all_data->thread.thread_levels[t].push_back(level);
            all_data->thread.level_threads[level].push_back(t);
            all_data->thread.barrier_flags[level][t] = 0;
         }
         all_data->thread.barrier_root[level] = 0;
      }
   }
}

void PartitionGrids(AllData *all_data)
{
   int num_level_threads;
   int num_levels =  all_data->grid.num_levels;

   if (all_data->input.thread_part_type == ALL_LEVELS){
      all_data->thread.A_ns = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.A_ne = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.R_ns = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.R_ne = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.P_ns = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.P_ne = (int **)malloc(num_levels * sizeof(int *));
     
      for (int level = 0; level < num_levels; level++){
         all_data->thread.A_ns[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.A_ne[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.R_ns[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.R_ne[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.P_ns[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.P_ne[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
      }

      for (int level = 0; level < num_levels; level++){
         num_level_threads = all_data->thread.level_threads[level].size();
        // printf("level %d/%d:", level, num_levels-1);
         for (int inner_level = 0; inner_level < level+2; inner_level++){
            if (inner_level < num_levels){
              // printf("\n\tlevel %d, n = %d: ", inner_level, all_data->grid.n[inner_level]);
               for (int i = 0; i < all_data->thread.level_threads[level].size(); i++){
                  int n = all_data->grid.n[inner_level];
                  int t = all_data->thread.level_threads[level][i];
     
                  int shift_t = t - all_data->thread.level_threads[level][0];
                  int size = n/num_level_threads;
                  int rest = n - size*num_level_threads;
                  if (shift_t < rest){
                     all_data->thread.A_ns[inner_level][t] = shift_t*size + shift_t;
                     all_data->thread.A_ne[inner_level][t] = (shift_t + 1)*size + shift_t + 1;
                  }
                  else{
                     all_data->thread.A_ns[inner_level][t] = shift_t*size + rest;
                     all_data->thread.A_ne[inner_level][t] = (shift_t + 1)*size + rest;
                  }
                  if (inner_level < num_levels-1){
                     n = hypre_CSRMatrixNumRows(all_data->matrix.P[inner_level]);
                     size = n/num_level_threads;
                     rest = n - size*num_level_threads;
                     if (shift_t < rest){
                        all_data->thread.P_ns[inner_level][t] = shift_t*size + shift_t;
                        all_data->thread.P_ne[inner_level][t] = (shift_t + 1)*size + shift_t + 1;
                     }
                     else{
                        all_data->thread.P_ns[inner_level][t] = shift_t*size + rest;
                        all_data->thread.P_ne[inner_level][t] = (shift_t + 1)*size + rest;
                     }
 
                     n = hypre_CSRMatrixNumRows(all_data->matrix.R[inner_level]);
                     size = n/num_level_threads;
                     rest = n - size*num_level_threads;
                     if (shift_t < rest){
                        all_data->thread.R_ns[inner_level][t] = shift_t*size + shift_t;
                        all_data->thread.R_ne[inner_level][t] = (shift_t + 1)*size + shift_t + 1;
                     }
                     else{
                        all_data->thread.R_ns[inner_level][t] = shift_t*size + rest;
                        all_data->thread.R_ne[inner_level][t] = (shift_t + 1)*size + rest;
                     }
                  }
                 // printf(" (%d,%d,%d)", shift_t, all_data->thread.A_ns[inner_level][t], all_data->thread.A_ne[inner_level][t]);
               }
            }
         }
        // printf("\n");
      } 
   }
   else{
      all_data->thread.A_ns = (int **)malloc(all_data->grid.num_levels * sizeof(int *));
      all_data->thread.A_ne = (int **)malloc(all_data->grid.num_levels * sizeof(int *));
      for (int level = 0; level < all_data->grid.num_levels; level++){
         int n = all_data->grid.n[level];

         all_data->thread.A_ns[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.A_ne[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));

         int num_threads = all_data->input.num_threads;
        // printf("level %d/%d:", level, num_levels-1);
         for (int t = 0; t < num_threads; t++){
            int size = n/num_threads;
            int rest = n - size*num_threads;
            if (t < rest)
            {
               all_data->thread.A_ns[level][t] = t*size + t;
               all_data->thread.A_ne[level][t] = (t + 1)*size + t + 1;
            }
            else
            {
               all_data->thread.A_ns[level][t] = t*size + rest;
               all_data->thread.A_ne[level][t] = (t + 1)*size + rest;
            }
           // printf(" (%d,%d,%d)", t, all_data->thread.A_ns[level][t], all_data->thread.A_ne[level][t]);
         }
        // printf("\n");
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
