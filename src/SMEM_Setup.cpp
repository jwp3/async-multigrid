#include "Main.hpp"
#include "SMEM_Setup.hpp"
#include "Misc.hpp"
#include "../eigen/Eigen/Sparse"

using namespace std;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor> EigenSpMat;
typedef Eigen::Triplet<double> EigenTrip;
typedef Eigen::VectorXd EigenVec;

void SmoothTransfer(AllData *all_data,
                    hypre_CSRMatrix *P,
                    hypre_CSRMatrix *R,
                    int level);

void EigenMatMat(AllData *all_data,
                 hypre_CSRMatrix *A,
                 hypre_CSRMatrix *B,
                 hypre_CSRMatrix *C);

void InitAlgebra(void *amg_vdata,
                 AllData *all_data);

void ComputeWork(AllData *all_data);

void PartitionLevels(AllData *all_data);

void PartitionGrids(AllData *all_data);

void StdVector_to_CSR(hypre_CSRMatrix *A,
                      vector<vector<HYPRE_Int>> j_vector,
                      vector<vector<HYPRE_Real>> data_vector);

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
         if (all_data->input.solver == MULTADD ||
             all_data->input.solver == ASYNC_MULTADD){
            all_data->matrix.P[level] = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
            all_data->matrix.R[level] = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
            SmoothTransfer(all_data,
                           hypre_ParCSRMatrixDiag(parP[level]),
                           hypre_ParCSRMatrixDiag(parR[level]),
                           level);
         }
         else {
            all_data->matrix.P[level] = hypre_ParCSRMatrixDiag(parP[level]);
            all_data->matrix.R[level] = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
            CSR_Transpose(hypre_ParCSRMatrixDiag(parR[level]), all_data->matrix.R[level]);
         }
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

   ComputeWork(all_data);

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

   all_data->thread.thread_levels.resize(num_threads, vector<int>(0));
   all_data->thread.level_threads.resize(num_levels, vector<int>(0));
   all_data->thread.barrier_flags = (int **)malloc(num_levels * sizeof(int *));
   all_data->thread.barrier_root = (int *)malloc(num_levels * sizeof(int));
   all_data->thread.loc_sum = (double *)malloc(num_threads * sizeof(double));

   all_data->output.smooth_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.residual_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.restrict_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.prolong_wtime = (double *)malloc(num_threads * sizeof(double));
  
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
      else if (all_data->input.thread_part_distr_type == EQUAL_THREADS){
         int equal_threads = max(all_data->input.num_threads/all_data->grid.num_levels, 1);
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
      else{
         for (int level = 0; level < num_levels; level++){
            int balanced_threads;
            if (num_threads == 1){
              // printf("%d ", tid);
               all_data->thread.thread_levels[tid].push_back(level);
               all_data->thread.level_threads[level].push_back(tid);
               all_data->thread.barrier_root[level] = tid;
               all_data->thread.barrier_flags[level][tid] = 0;
            }
            else{
              // if (level == 0){
              //    balanced_threads = max((int)ceil(all_data->thread.frac_level_work[level] *
              //                                         (double)all_data->input.num_threads), 1);
              // }
              // else if (level == num_levels-1){
               
	       if (level == num_levels-1){
                  balanced_threads = num_threads;
               }
               else {
                  balanced_threads = max((int)ceil(all_data->thread.frac_level_work[level] *
                                                        (double)all_data->input.num_threads), 1);
                  while (balanced_threads >= num_threads){
                     balanced_threads--;
                  }
                  while (1){
                     int candidate = balanced_threads-1;
                     double diff_current =
                        fabs(all_data->thread.frac_level_work[level] -
                             (double)balanced_threads/(double)all_data->input.num_threads);
                     double diff_candidate =
                        fabs(all_data->thread.frac_level_work[level] -
                             (double)candidate/(double)all_data->input.num_threads);
                     if (diff_current <= diff_candidate){
                        break;
                     }
                     balanced_threads--;
                  }
                 // balanced_threads = max((int)floor(all_data->thread.frac_level_work[level] *
                 //                                       (double)all_data->input.num_threads), 1);
                 // while (1){
                 //    int candidate = balanced_threads+1;
                 //    double diff_current =
                 //       fabs(all_data->thread.frac_level_work[level] -
                 //            (double)balanced_threads/(double)all_data->input.num_threads);
                 //    double diff_candidate =
                 //       fabs(all_data->thread.frac_level_work[level] -
                 //            (double)candidate/(double)all_data->input.num_threads);
                 //    if (diff_current <= diff_candidate ||
                 //        (double)candidate/(double)all_data->input.num_threads > all_data->thread.frac_level_work[level]){
                 //       break;
                 //    }
                 //    balanced_threads++;
                 // }
               }
               for (int t = tid; t < tid + balanced_threads; t++){
                 // printf("%d ", t);
                  all_data->thread.thread_levels[t].push_back(level);
                  all_data->thread.level_threads[level].push_back(t);
               }
               for (int t = tid; t < tid + balanced_threads; t++){
                  all_data->thread.barrier_flags[level][t] = 0;
               }
               num_threads -= balanced_threads;
               tid += balanced_threads;
               all_data->thread.barrier_root[level] = tid-1;
            }
	    printf("\tlevel %d: %f, %f\n",
                      level,
                      all_data->thread.frac_level_work[level],
                      (double)balanced_threads/(double)all_data->input.num_threads);
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

void ComputeWork(AllData *all_data)
{
   int coarsest_level;
   int fine_grid, coarse_grid;
   all_data->thread.level_work = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->thread.level_work[level] = hypre_CSRMatrixNumNonzeros(all_data->matrix.A[0]);
      if (all_data->input.solver == MULTADD ||
          all_data->input.solver == ASYNC_MULTADD){
         coarsest_level = level;
      }
      else if (all_data->input.solver == AFACX ||
               all_data->input.solver == ASYNC_AFACX){
         coarsest_level = level+1;
      }
      for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
         fine_grid = inner_level;
         coarse_grid = inner_level + 1;
         if (all_data->input.solver == MULTADD ||
             all_data->input.solver == ASYNC_MULTADD){
            all_data->thread.level_work[level] +=
               hypre_CSRMatrixNumNonzeros(all_data->matrix.R[fine_grid]);

         }
         else if (all_data->input.solver == AFACX ||
                  all_data->input.solver == ASYNC_AFACX){
            if (level < all_data->grid.num_levels-1){
               all_data->thread.level_work[level] +=
                  hypre_CSRMatrixNumNonzeros(all_data->matrix.R[fine_grid]);
            }
         }
      }

      fine_grid = level;
      coarse_grid = level + 1;
      if (level == all_data->grid.num_levels-1){
         all_data->thread.level_work[level] += hypre_CSRMatrixNumNonzeros(all_data->matrix.A[fine_grid]);
      }
      else {
         if (all_data->input.solver == MULTADD ||
             all_data->input.solver == ASYNC_MULTADD){
            all_data->thread.level_work[level] +=
	       all_data->input.num_fine_smooth_sweeps * hypre_CSRMatrixNumNonzeros(all_data->matrix.A[fine_grid]);
         }
         else if (all_data->input.solver == AFACX ||
                  all_data->input.solver == ASYNC_AFACX){
            all_data->thread.level_work[level] +=
               all_data->input.num_coarse_smooth_sweeps * hypre_CSRMatrixNumNonzeros(all_data->matrix.A[coarse_grid]) +
               hypre_CSRMatrixNumNonzeros(all_data->matrix.P[fine_grid]) +
	       hypre_CSRMatrixNumNonzeros(all_data->matrix.A[fine_grid]) +
               all_data->input.num_fine_smooth_sweeps * hypre_CSRMatrixNumNonzeros(all_data->matrix.A[fine_grid]);
         }
      }

      coarsest_level = level;
      for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
         fine_grid = inner_level;
         coarse_grid = inner_level + 1;
         if (all_data->input.solver == MULTADD ||
             all_data->input.solver == ASYNC_MULTADD){
            all_data->thread.level_work[level] +=
               hypre_CSRMatrixNumNonzeros(all_data->matrix.P[fine_grid]);

         }
         else if (all_data->input.solver == AFACX ||
                  all_data->input.solver == ASYNC_AFACX){
            all_data->thread.level_work[level] +=
               hypre_CSRMatrixNumNonzeros(all_data->matrix.P[fine_grid]);
         }
      }
   }
   all_data->thread.tot_work = 0;
   all_data->thread.frac_level_work = (double *)calloc(all_data->grid.num_levels, sizeof(double));
   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->thread.tot_work += all_data->thread.level_work[level];
   }
   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->thread.frac_level_work[level] = (double)all_data->thread.level_work[level] / (double)all_data->thread.tot_work;
     // printf("hello %e\n", all_data->thread.frac_level_work[level]);
   }
}

void SmoothTransfer(AllData *all_data,
                    hypre_CSRMatrix *P,
                    hypre_CSRMatrix *R,
                    int level)
{
   hypre_CSRMatrix *RT = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
   CSR_Transpose(R, RT);

   hypre_CSRMatrix *G = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
   hypre_CSRMatrix *GT = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));

   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(all_data->matrix.A[level]);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(all_data->matrix.A[level]);
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(all_data->matrix.A[level]);

   hypre_CSRMatrixNumRows(G) = num_rows;
   hypre_CSRMatrixNumCols(G) = num_cols;
   hypre_CSRMatrixNumNonzeros(G) = nnz;

   hypre_CSRMatrixNumRows(GT) = num_rows;
   hypre_CSRMatrixNumCols(GT) = num_cols;
   hypre_CSRMatrixNumNonzeros(GT) = nnz;

   HYPRE_Int *A_i = hypre_CSRMatrixI(all_data->matrix.A[level]);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(all_data->matrix.A[level]);
   HYPRE_Real *A_data = hypre_CSRMatrixData(all_data->matrix.A[level]);
   HYPRE_Real *G_data = (HYPRE_Real *)calloc(nnz, sizeof(HYPRE_Real));
   HYPRE_Real *GT_data = (HYPRE_Real *)calloc(nnz, sizeof(HYPRE_Real));

   for (int i = 0; i < num_rows; i++){
      if (A_data[A_i[i]] != 0.0){
         G_data[A_i[i]] = 1.0 - all_data->input.smooth_weight;
         GT_data[A_i[i]] = 1.0 - all_data->input.smooth_weight;
         for (int jj = A_i[i]+1; jj < A_i[i+1]; jj++){
            G_data[jj] = -all_data->input.smooth_weight * A_data[jj] / A_data[A_i[i]];
            GT_data[jj] = -all_data->input.smooth_weight * A_data[jj] / A_data[A_i[A_j[jj]]];
         }
      }
   }

   hypre_CSRMatrixI(G) = A_i;
   hypre_CSRMatrixJ(G) = A_j;
   hypre_CSRMatrixData(G) = G_data;
   hypre_CSRMatrixI(GT) = A_i;
   hypre_CSRMatrixJ(GT) = A_j;
   hypre_CSRMatrixData(GT) = GT_data;
   EigenMatMat(all_data, G, P, all_data->matrix.P[level]);
   EigenMatMat(all_data, RT, GT, all_data->matrix.R[level]);
}

void EigenMatMat(AllData *all_data,
                 hypre_CSRMatrix *A,
                 hypre_CSRMatrix *B,
                 hypre_CSRMatrix *C)
{
   Eigen::setNbThreads(all_data->input.num_threads);

   int A_num_rows = hypre_CSRMatrixNumRows(A); 
   int A_num_cols = hypre_CSRMatrixNumCols(A);
   int B_num_rows = hypre_CSRMatrixNumRows(B);
   int B_num_cols = hypre_CSRMatrixNumCols(B);
   
   EigenSpMat eigen_A(A_num_rows, A_num_cols);
   EigenSpMat eigen_B(B_num_rows, B_num_cols);
   EigenSpMat eigen_C(A_num_rows, B_num_cols);
 
   vector<EigenTrip> A_eigtrip;
   vector<EigenTrip> B_eigtrip;

   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);

   HYPRE_Int *B_i = hypre_CSRMatrixI(B);
   HYPRE_Int *B_j = hypre_CSRMatrixJ(B);
   HYPRE_Real *B_data = hypre_CSRMatrixData(B);
   

   for (int i = 0; i < A_num_rows; i++){
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         A_eigtrip.push_back(EigenTrip(i, A_j[jj], A_data[jj]));
      }
   }
   for (int i = 0; i < B_num_rows; i++){
      for (int jj = B_i[i]; jj < B_i[i+1]; jj++){
         B_eigtrip.push_back(EigenTrip(i, B_j[jj], B_data[jj]));
      }
   }

   eigen_A.setFromTriplets(A_eigtrip.begin(), A_eigtrip.end());
   eigen_B.setFromTriplets(B_eigtrip.begin(), B_eigtrip.end());
   eigen_A.makeCompressed();
   eigen_B.makeCompressed();
   eigen_C = (eigen_A * eigen_B).pruned();
   eigen_C.makeCompressed();

   int num_rows, num_cols, nnz;
   hypre_CSRMatrixNumRows(C) = num_rows = eigen_C.rows();
   hypre_CSRMatrixNumCols(C) = num_cols = eigen_C.cols();
   hypre_CSRMatrixNumNonzeros(C) = nnz = eigen_C.nonZeros();

   vector<vector<HYPRE_Int>>
      j_vector(num_rows, vector<HYPRE_Int>(0));
   vector<vector<HYPRE_Real>>
      data_vector(num_rows, vector<HYPRE_Real>(0));

   for (int i = 0; i < eigen_C.rows(); i++){
      for (EigenSpMat::InnerIterator it(eigen_C,i); it; ++it) {
         j_vector[i].push_back(it.col());
         data_vector[i].push_back(it.value());
      }
   }

   StdVector_to_CSR(C, j_vector, data_vector);
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

   vector<vector<HYPRE_Int>>
      j_vector(A_num_cols, vector<HYPRE_Int>(0));
   vector<vector<HYPRE_Real>>
      data_vector(A_num_cols, vector<HYPRE_Real>(0));

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
                      vector<vector<HYPRE_Int>> j_vector,
                      vector<vector<HYPRE_Real>> data_vector)
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
