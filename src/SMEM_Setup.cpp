#include "Main.hpp"
#include "SMEM_Setup.hpp"
#include "Misc.hpp"
#include "Laplacian.hpp"
#include "Elasticity.hpp"
#include "Maxwell.hpp"
#include "SMEM_Cheby.hpp"
#include "Sparse"
#include "Cholesky"
#include "BuildHypreMatrix.hpp"
#include "BuildMfemMatrix.hpp"

//TODO: replace PARDISO and EIGEN

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
void InitAlgebra(AllData *all_data);
void ComputeWork(AllData *all_data);
void PartitionLevels(AllData *all_data);
void PartitionGrids(AllData *all_data);
void StdVector_to_CSR(hypre_CSRMatrix *A,
                      vector<vector<HYPRE_Int>> j_vector,
                      vector<vector<HYPRE_Real>> data_vector);
void CSR_Transpose(hypre_CSRMatrix *A,
                   hypre_CSRMatrix *AT);
void SetupThreads(AllData *all_data);
void CSR_to_Vector(AllData *all_data,
                   hypre_CSRMatrix *A,
                   vector<vector<HYPRE_Int>> *col_vec,
                   vector<vector<HYPRE_Real>> *val_vec,
                   int row_start, int col_start);
void BuildExtendedMatrix(AllData *all_data,
                         hypre_CSRMatrix **A_array,
                         hypre_CSRMatrix **P_array,
                         hypre_CSRMatrix **R_array,
                         hypre_CSRMatrix **B_ptr);
void SMEM_CSRMatrixGetLoadBalancedPartitionBoundary(AllData *all_data,
                                                    hypre_CSRMatrix *A,
                                                    int size_per_thread,
                                                    int num_threads, int t,
                                                    int *ns, int *ne);

void SMEM_Setup(AllData *all_data)
{
   double start;
   int num_threads = all_data->input.num_threads;
   start = omp_get_wtime();
   SMEM_BuildMatrix(all_data);
   all_data->output.prob_setup_wtime = omp_get_wtime() - start;

   start = omp_get_wtime();
   SMEM_SetHypreParameters(all_data);
   HYPRE_BoomerAMGSetup(all_data->hypre.solver,
                        all_data->hypre.parcsr_A,
                        all_data->hypre.par_b,
                        all_data->hypre.par_x);
   all_data->output.hypre_setup_wtime = omp_get_wtime() - start;

   //HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, 50);
   //HYPRE_BoomerAMGSolve(all_data->hypre.solver,
   //                     all_data->hypre.parcsr_A,
   //                     all_data->hypre.par_b,
   //                     all_data->hypre.par_x);
   //exit(1);

   if (all_data->input.hypre_solver_flag == 1) return;
   
   if (all_data->input.format_output_flag == 0){
      printf("\n\nhypre setup time %e\n\n", all_data->output.hypre_setup_wtime);
   }

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   all_data->grid.num_levels = (int)hypre_ParAMGDataNumLevels(amg_data);

   //char buffer[100];
   //hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   //hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   //hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   //for (int level = 0; level < all_data->grid.num_levels; level++){
   //   if (level < all_data->grid.num_levels-1){
   //      sprintf(buffer, "P_%d.txt", level+1);
   //      PrintCSRMatrix(hypre_ParCSRMatrixDiag(P_array[level]), buffer, 0);
   //      sprintf(buffer, "R_%d.txt", level+1);
   //      PrintCSRMatrix(hypre_ParCSRMatrixDiag(R_array[level]), buffer, 0);
   //   }
   //   sprintf(buffer, "A_%d.txt", level+1);
   //   PrintCSRMatrix(hypre_ParCSRMatrixDiag(A_array[level]), buffer, 0);
   //}
  // return;


   all_data->barrier.local_sense = (int *)calloc(num_threads, sizeof(int));
  // all_data->barrier.counter = (int *)calloc(all_data->grid.num_levels, sizeof(int));
  // all_data->barrier.flag = (int *)calloc(all_data->grid.num_levels, sizeof(int));

   all_data->grid.global_smooth_flags = (int *)calloc(num_threads, sizeof(int));
   all_data->grid.zero_flags = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   all_data->grid.num_smooth_wait = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   all_data->grid.finest_num_res_compute = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   all_data->grid.local_num_res_compute = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX || all_data->input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
      all_data->grid.local_num_correct = (int *)calloc(all_data->input.num_threads, sizeof(int));
   }
   else {
      all_data->grid.local_num_correct = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   }
   all_data->grid.local_cycle_num_correct = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   all_data->grid.last_read_correct = (int *)calloc(all_data->grid.num_levels, sizeof(int));
   all_data->grid.last_read_cycle_correct = (int *)calloc(all_data->grid.num_levels, sizeof(int));

   all_data->grid.mean_grid_wait = (double *)calloc(all_data->grid.num_levels, sizeof(double));
   all_data->grid.max_grid_wait = (double *)calloc(all_data->grid.num_levels, sizeof(double));
   all_data->grid.min_grid_wait = (double *)calloc(all_data->grid.num_levels, sizeof(double));

   all_data->output.smooth_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.residual_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.restrict_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.prolong_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.A_matvec_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.vec_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.innerprod_wtime = (double *)malloc(num_threads * sizeof(double));
   all_data->output.smooth_sweeps = (int *)malloc(num_threads * sizeof(int));

   if (all_data->input.solver != EXPLICIT_EXTENDED_SYSTEM_BPX){
      start = omp_get_wtime();
      hypre_GaussElimSetup(amg_data, all_data->grid.num_levels-1, 99);
      //printf("GaussElim %e\n", omp_get_wtime() - start);
   }

   start = omp_get_wtime();
   InitAlgebra(all_data);
   //printf("InitAlgebra %e\n", omp_get_wtime() - start);

   if (all_data->input.solver != EXPLICIT_EXTENDED_SYSTEM_BPX){
      start = omp_get_wtime();
      ComputeWork(all_data);
      //printf("ComputeWork %e\n", omp_get_wtime() - start);
      start = omp_get_wtime();
      PartitionLevels(all_data);
      //printf("PartitionLevels %e\n", omp_get_wtime() - start);

      start = omp_get_wtime();
      PartitionGrids(all_data);
      //printf("InitAlgebra %e\n", omp_get_wtime() - start);
   }


   if (all_data->input.thread_part_type == ALL_LEVELS && all_data->input.construct_R_flag == 0){
      for (int level = 0; level < all_data->grid.num_levels; level++){
         int num_level_threads = all_data->thread.level_threads[level].size();
         int coarsest_level = all_data->grid.num_levels;
         all_data->level_vector[level].y_expand = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         for (int inner_level = 0; inner_level < coarsest_level-1; inner_level++){
            int num_cols = hypre_CSRMatrixNumCols(all_data->matrix.R[inner_level]);
            all_data->level_vector[level].y_expand[inner_level] = (HYPRE_Real *)calloc(num_level_threads * num_cols, sizeof(HYPRE_Real));
         }
      }
   }

   start = omp_get_wtime();
   if (all_data->input.cheby_flag == 1){
      ChebySetup(all_data);
   }
   //printf("Cheby %e\n", omp_get_wtime() - start);

   all_data->output.num_cycles = 0;

}

void InitAlgebra(AllData *all_data)
{
   double start;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   int num_threads = all_data->input.num_threads;
   
   hypre_ParCSRMatrix **parA;
   hypre_ParCSRMatrix **parP;
   hypre_ParCSRMatrix **parR;

   parA = hypre_ParAMGDataAArray(amg_data);
   parP = hypre_ParAMGDataPArray(amg_data);
   parR = hypre_ParAMGDataRArray(amg_data);
   
   all_data->grid.n = (int *)malloc(all_data->grid.num_levels * sizeof(int));

   all_data->vector.i.resize(all_data->grid.num_levels, vector<int>(0));

   all_data->matrix.A =
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix *));
   all_data->matrix.P =
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix *));
   all_data->matrix.R =
      (hypre_CSRMatrix **)malloc(all_data->grid.num_levels * sizeof(hypre_CSRMatrix *));
   if (all_data->input.smoother == L1_JACOBI ||
       all_data->input.smoother == ASYNC_L1_JACOBI ||
       all_data->input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL){
      all_data->matrix.L1_row_norm = (double **)malloc(all_data->grid.num_levels * sizeof(double *));
   }
   else {
      all_data->matrix.A_diag = (double **)malloc(all_data->grid.num_levels * sizeof(double *));
   }
   all_data->vector.y_expand = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));

   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->matrix.A[level] = hypre_ParCSRMatrixDiag(parA[level]);
      all_data->grid.n[level] = hypre_CSRMatrixNumRows(all_data->matrix.A[level]);
      HYPRE_Real *A_data = hypre_CSRMatrixData(all_data->matrix.A[level]);
      HYPRE_Int *A_i = hypre_CSRMatrixI(all_data->matrix.A[level]);
      if (all_data->input.smoother == L1_JACOBI ||
          all_data->input.smoother == ASYNC_L1_JACOBI ||
          all_data->input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL){
         all_data->matrix.L1_row_norm[level] = (double *)malloc(all_data->grid.n[level] * sizeof(double));
         for (int i = 0; i < all_data->grid.n[level]; i++){
            all_data->matrix.L1_row_norm[level][i] = 0;
            for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
               all_data->matrix.L1_row_norm[level][i] += fabs(A_data[jj]);
            }
         }
      }
      else {
         all_data->matrix.A_diag[level] = (double *)malloc(all_data->grid.n[level] * sizeof(double));
         for (int i = 0; i < all_data->grid.n[level]; i++){
            all_data->matrix.A_diag[level][i] = A_data[A_i[i]] / all_data->input.smooth_weight;
         }
      }
      all_data->vector.i[level].resize(all_data->grid.n[0]);
      for (int i = 0; i < all_data->grid.n[0]; i++){
         all_data->vector.i[level][i] = i;
      }
      if (level < all_data->grid.num_levels-1){
         if (all_data->input.solver == MULTADD ||
             all_data->input.solver == ASYNC_MULTADD){
            if (all_data->input.num_post_smooth_sweeps > 0){
               all_data->matrix.P[level] = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
	    }
	    else {
	       all_data->matrix.P[level] = hypre_ParCSRMatrixDiag(parP[level]);
	    }
            all_data->matrix.R[level] = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
	    if (all_data->input.num_pre_smooth_sweeps == 0){
               hypre_CSRMatrixTranspose(hypre_ParCSRMatrixDiag(parR[level]), &(all_data->matrix.R[level]), 1);
	      // CSR_Transpose(hypre_ParCSRMatrixDiag(parR[level]), all_data->matrix.R[level]);
	    }
            SmoothTransfer(all_data,
                           hypre_ParCSRMatrixDiag(parP[level]),
                           hypre_ParCSRMatrixDiag(parR[level]),
                           level);
         }
         else {
            all_data->matrix.P[level] = hypre_ParCSRMatrixDiag(parP[level]);
            if (all_data->input.construct_R_flag == 1){
               all_data->matrix.R[level] = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
               hypre_CSRMatrixTranspose(hypre_ParCSRMatrixDiag(parR[level]), &(all_data->matrix.R[level]), 1);
            }
            else {
               all_data->matrix.R[level] = all_data->matrix.P[level];
               int num_cols = hypre_CSRMatrixNumCols(all_data->matrix.R[level]);
               all_data->vector.y_expand[level] = (HYPRE_Real *)calloc(num_threads * num_cols, sizeof(HYPRE_Real));
            }
           // CSR_Transpose(hypre_ParCSRMatrixDiag(parR[level]), all_data->matrix.R[level]);
         }
      }
   }

   // TODO: cleanup excess memory usage
   if (all_data->input.solver != EXPLICIT_EXTENDED_SYSTEM_BPX){
      all_data->vector.u = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.y = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.r = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.f = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.u_prev = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.z = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.e = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.u_smooth = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
   
      if (all_data->input.thread_part_type == ALL_LEVELS/* &&
          all_data->input.num_threads > 1*/){
   
         all_data->level_vector =
            (VectorData *)malloc(all_data->grid.num_levels * sizeof(VectorData));
   
         for (int level = 0; level < all_data->grid.num_levels; level++){
   
            all_data->level_vector[level].f = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].u = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].u_prev = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].u_coarse = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].u_coarse_prev = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].u_fine = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].u_fine_prev = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].y = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].r = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].r_coarse = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].r_fine = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].e = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].z = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].z1 = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
            all_data->level_vector[level].z2 = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
         }
   
         for (int level = 0; level < all_data->grid.num_levels; level++){
            int coarsest_level;
            if (all_data->input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
               coarsest_level = all_data->grid.num_levels;
            }
            else {
               coarsest_level = level+2;
            }
            for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
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
                  all_data->level_vector[level].z[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
                  all_data->level_vector[level].z1[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
                  all_data->level_vector[level].z2[inner_level] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               }
            }
            int n = all_data->grid.n[level];
            if (all_data->input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
               all_data->vector.f[level] =
                  (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->vector.r[level] =
                  (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->vector.u[level] =
                  (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->vector.y[level] =
                  (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->vector.u_prev[level] =
                  (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->vector.z[level] =
                  (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->vector.e[level] =
                  (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
               all_data->vector.u_smooth[level] =
                  (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
            }
            else {
               if (level == 0){
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
      int level = 0;
      HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));
      for (int i = 0; i < all_data->grid.n[level]; i++){
         all_data->vector.f[level][i] = f_local_data[i];
      }

      if (all_data->input.solver == PAR_BPX) {
         all_data->grid.disp = (int *)malloc((all_data->grid.num_levels + 1) * sizeof(int));
         all_data->grid.N = 0;
         all_data->grid.disp[0] = 0;
         for (int level = 0; level < all_data->grid.num_levels; level++){
            int num_rows = all_data->grid.n[level];
            all_data->grid.disp[level+1] = all_data->grid.disp[level] + num_rows;
            all_data->grid.N += num_rows;
         } 
         all_data->vector.xx = (HYPRE_Real *)calloc(all_data->grid.N, sizeof(HYPRE_Real));
         all_data->vector.rr = (HYPRE_Real *)calloc(all_data->grid.N, sizeof(HYPRE_Real));

         if (all_data->input.smoother == L1_JACOBI ||
             all_data->input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL){
            all_data->matrix.L1_row_norm_ext = (HYPRE_Real *)calloc(all_data->grid.N, sizeof(HYPRE_Real));
            int k = 0;
            for (int level = 0; level < all_data->grid.num_levels; level++){
               int num_rows = all_data->grid.n[level];
               for (int i = 0; i < num_rows; i++){
                  all_data->matrix.L1_row_norm_ext[k] = all_data->matrix.L1_row_norm[level][i];
                  k++;
               }
            }
         }
         else {
            all_data->matrix.A_diag_ext = (HYPRE_Real *)calloc(all_data->grid.N, sizeof(HYPRE_Real));
            int k = 0;
            for (int level = 0; level < all_data->grid.num_levels; level++){
               int num_rows = all_data->grid.n[level];
               HYPRE_Int *A_i = hypre_CSRMatrixI(all_data->matrix.A[level]);
               HYPRE_Real *A_data = hypre_CSRMatrixData(all_data->matrix.A[level]);
               for (int i = 0; i < num_rows; i++){
                  all_data->matrix.A_diag_ext[k] = A_data[A_i[i]] / all_data->input.smooth_weight;
                  k++;
               }
            }
         }
      }
      //if (all_data->input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
      //   double start = omp_get_wtime();
      //   BuildExtendedMatrix(all_data,
      //                       all_data->matrix.A,
      //                       all_data->matrix.P,
      //                       all_data->matrix.R,
      //                       &(all_data->matrix.AA));
      //   double stop = omp_get_wtime() - start;

      //   int ns, ne;
      //   int nnz = hypre_CSRMatrixNumNonzeros(all_data->matrix.AA);
      //   int nt = 272;
      //   int nnz_per_thread = (nnz + nt - 1)/nt;
      //   int disp = all_data->grid.disp[1];
      //   int level = 0, t_count = 0;
      //   for (int t = 0; t < nt; t++){
      //      SMEM_CSRMatrixGetLoadBalancedPartitionBoundary(all_data, all_data->matrix.AA, nnz_per_thread, nt, t, &ns, &ne);
      //      if (ns > disp){
      //         printf("%d %d %d %d %d\n", level, t_count, ns, disp, all_data->grid.disp[level+2]);
      //         t_count = 0;
      //         level++;
      //         disp = all_data->grid.disp[level+1];
      //      }
      //      t_count++;
      //      //printf("%d %d %d %d %d\n", t, level, ns, ne, all_data->grid.disp[7]);
      //   }
      //   printf("%d %d\n", level, t_count);

      //   char buffer[100] = "matlab_code/AA.txt";
      //   PrintCSRMatrix(all_data->matrix.AA, buffer, 0);
      //}
   }
   else {
      hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
      hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
      double start = omp_get_wtime();
      BuildExtendedMatrix(all_data,
                          all_data->matrix.A,
                          all_data->matrix.P,
                          all_data->matrix.R,
                          &(all_data->matrix.AA));
      double stop = omp_get_wtime() - start;
      if (all_data->input.format_output_flag == 0){
         printf("\nExtended matrix constructed, time %e\n", stop);
      }
      int N = hypre_CSRMatrixNumRows(all_data->matrix.AA);
      int NNZ = hypre_CSRMatrixNumNonzeros(all_data->matrix.AA);
      all_data->vector.xx = (HYPRE_Real *)calloc(N, sizeof(HYPRE_Real));
      all_data->vector.xx_prev = (HYPRE_Real *)calloc(N, sizeof(HYPRE_Real));
      all_data->vector.yy = (HYPRE_Real *)calloc(N, sizeof(HYPRE_Real));
      all_data->vector.bb = (HYPRE_Real *)malloc(N * sizeof(HYPRE_Real));
      all_data->vector.rr = (HYPRE_Real *)malloc(N * sizeof(HYPRE_Real));
      all_data->vector.zz = (HYPRE_Real *)malloc(N * sizeof(HYPRE_Real));


      int *disp = all_data->grid.disp;
      int n = hypre_ParCSRMatrixNumRows(A_array[0]);
      HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));
      for (int i = 0; i < n; i++){
         all_data->vector.bb[disp[0]+i] = f_local_data[i];
      }
      for (int level = 0; level < all_data->grid.num_levels-1; level++){
         int fine_grid = level;
         int coarse_grid = level + 1;

         hypre_ParCSRMatrixMatvecT(1.0,
                                   R_array[fine_grid],
                                   F_array[fine_grid],
                                   0.0,
                                   F_array[coarse_grid]);

         n = hypre_ParCSRMatrixNumRows(A_array[coarse_grid]);
         f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[coarse_grid]));
         for (int i = 0; i < n; i++){
            all_data->vector.bb[disp[coarse_grid]+i] = f_local_data[i];
         }
      }

      n = hypre_ParCSRMatrixNumRows(A_array[0]);
      all_data->vector.y = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.y[0] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      all_data->vector.e = (HYPRE_Real **)malloc(all_data->grid.num_levels * sizeof(HYPRE_Real *));
      all_data->vector.e[0] = (HYPRE_Real *)calloc(n, sizeof(HYPRE_Real));
      
      all_data->thread.AA_NS = (int *)calloc(all_data->input.num_threads, sizeof(int));
      all_data->thread.AA_NE = (int *)calloc(all_data->input.num_threads, sizeof(int));
      int NNZ_per_thread = (NNZ + all_data->input.num_threads - 1)/all_data->input.num_threads;
      for (int t = 0; t < all_data->input.num_threads; t++){
         SMEM_CSRMatrixGetLoadBalancedPartitionBoundary(all_data,
                                                        all_data->matrix.AA,
                                                        NNZ_per_thread, 
                                                        all_data->input.num_threads,
                                                        t,
                                                        &(all_data->thread.AA_NS[t]), &(all_data->thread.AA_NE[t]));
      }
     // for (int t = 0; t < all_data->input.num_threads; t++){
     //    printf("%d %d %d %d\n", t, all_data->thread.AA_NS[t], all_data->thread.AA_NE[t],  NNZ_vec[t]);
     // }
     // if (all_data->input.num_threads > 1){
     //    idx_t nparts = (idx_t)all_data->input.num_threads;
     //    idx_t options[METIS_NOPTIONS];
     //    idx_t ncon = 1, objval;
     //    METIS_SetDefaultOptions(options);
     //    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
     //    idx_t *perm = (idx_t *)calloc(N, sizeof(idx_t));
     //    idx_t *xadj = hypre_CSRMatrixI(all_data->matrix.AA);
     //    idx_t *adjncy = hypre_CSRMatrixJ(all_data->matrix.AA);
     //    int metis_flag = METIS_OK;
     //    metis_flag = METIS_PartGraphKway(&N, &ncon, xadj, adjncy, NULL,
     //                                     NULL, NULL, &nparts, NULL, NULL,
     //                                     options, &objval, perm);
     //    if (metis_flag != METIS_OK){
     //       printf("****WARNING****: METIS returned error with code %d.\n", metis_flag);
     //    }
     // }
      
     // char buffer[100];
     // sprintf(buffer, "AA.txt");
     // PrintCSRMatrix(all_data->matrix.AA, buffer, 0);
      
   }
   //char buffer[100];
   //sprintf(buffer, "A.txt");
   //PrintCSRMatrix(all_data->matrix.A[0], buffer, 0);
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
   all_data->thread.global_barrier_flags = (int *)calloc(num_threads, sizeof(int));
   all_data->thread.loc_sum = (double *)malloc(num_threads * sizeof(double));
  
   if (all_data->input.thread_part_type == ALL_LEVELS){
      int tid = 0;
      for (int level = 0; level < num_levels; level++){
         all_data->thread.barrier_flags[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
      }
      int finest_level;
      if (all_data->input.res_compute_type == GLOBAL){
         finest_level = 1;
      }
      else{
         finest_level = 0;
      }
      if (all_data->input.thread_part_distr_type == HALF_THREADS){
         for (int level = finest_level; level < num_levels; level++){
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
         for (int level = finest_level; level < num_levels; level++){
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
         //for (int level = finest_level; level < num_levels; level++){
         //   int balanced_threads;
         //   if (num_threads == 1){
         //     // printf("%d ", tid);
         //      all_data->thread.thread_levels[tid].push_back(level);
         //      all_data->thread.level_threads[level].push_back(tid);
         //      all_data->thread.barrier_root[level] = tid;
         //      all_data->thread.barrier_flags[level][tid] = 0;
         //   }
         //   else{ 
	 //      if (level == num_levels-1 || num_threads == 1){
         //         balanced_threads = num_threads;
         //      }
         //      else {
         //        // balanced_threads = max((int)ceil(all_data->grid.frac_level_work[level] *
         //        //                                       (double)all_data->input.num_threads), 1);
         //        // while (balanced_threads >= num_threads){
         //        //    balanced_threads--;
         //        // }
         //        // while (1){
         //        //    int candidate = balanced_threads-1;
         //        //    double diff_current =
         //        //       fabs(all_data->grid.frac_level_work[level] -
         //        //            (double)balanced_threads/(double)all_data->input.num_threads);
         //        //    double diff_candidate =
         //        //       fabs(all_data->grid.frac_level_work[level] -
         //        //            (double)candidate/(double)all_data->input.num_threads);
         //        //    if (diff_current <= diff_candidate ||
         //        //        balanced_threads == 1){
         //        //       break;
         //        //    }
         //        //    balanced_threads--;
         //        // }

         //         balanced_threads = max((int)floor(all_data->grid.frac_level_work[level] *
         //                                               (double)all_data->input.num_threads), 1);
         //         while (1){
         //            int candidate = balanced_threads+1;
         //            double diff_current =
         //               fabs(all_data->grid.frac_level_work[level] -
         //                    (double)balanced_threads/(double)all_data->input.num_threads);
         //            double diff_candidate =
         //               fabs(all_data->grid.frac_level_work[level] -
         //                    (double)candidate/(double)all_data->input.num_threads);
         //            if (diff_current <= diff_candidate ||
         //                (double)candidate/(double)all_data->input.num_threads > all_data->grid.frac_level_work[level]){
         //               break;
         //            }
         //            balanced_threads++;
         //         }
         //      }
         //      for (int t = tid; t < tid + balanced_threads; t++){
         //        // printf("%d ", t);
         //         all_data->thread.thread_levels[t].push_back(level);
         //         all_data->thread.level_threads[level].push_back(t);
         //      }
         //      for (int t = tid; t < tid + balanced_threads; t++){
         //         all_data->thread.barrier_flags[level][t] = 0;
         //      }
         //      num_threads -= balanced_threads;
	 //      all_data->thread.barrier_root[level] = tid;
         //      tid += balanced_threads;
         //   }
	 //   //printf("\tlevel %d: %f, %f, %d\n",
         //   //       level,
         //   //       all_data->grid.frac_level_work[level],
         //   //       (double)balanced_threads/(double)all_data->input.num_threads,
         //   //       balanced_threads);
         //   //printf("\n");
         //}

         vector<int> threads_per_level(num_levels, 0);
         int level = 0;
         for (int t = 0; t < num_threads; t++){
            threads_per_level[level]++;
            level = level < num_levels-1 ? level+1 : 0;
         }

         double prev_max_diff = DBL_MAX;
         while(1){
            double max_diff = 0.0;
            int k_max = 0;
            /* first find max difference */
            for (int k = 0; k < num_levels; k++){
               double frac = (double)threads_per_level[k]/(double)num_threads;
               double diff = all_data->grid.frac_level_work[k] - frac;
               if (fabs(diff) > fabs(max_diff)){
                  if (max_diff < 0.0 && threads_per_level[k] == 1){
                  }
                  else {
                     max_diff = diff;
                     k_max = k;
                  }
               }
            }
            double min_diff = DBL_MAX;
            int k_min = 0;
            int min_not_found = 1;
            /* find min difference based on result from max */
            for (int k = 0; k < num_levels; k++){
               if (k != k_max){
                  double frac = (double)threads_per_level[k]/(double)num_threads;
                  double diff = all_data->grid.frac_level_work[k] - frac;
                  if (max_diff > 0.0){ /* if we need to increase the number of threads */
                     if (diff < 0.0 &&  /* we need to find another level that decreases the number of threads */
                         threads_per_level[k] > 1 && /* target level must have at least two threads */
                         fabs(diff) < fabs(min_diff)){ /*now we can compute min */
                        min_diff = diff;
                        k_min = k;
                        min_not_found = 0;
                     }
                  }
                  else { /* if we need to decrease number of threads */ 
                     if (diff > 0.0 &&
                         fabs(diff) < fabs(min_diff)){
                        min_diff = diff;
                        k_min = k;
                        min_not_found = 0;
                     }
                  }
               }
            }
            //printf("%e %e\n", max_diff, prev_max_diff);
            if (fabs(max_diff) >= fabs(prev_max_diff) || min_not_found){
               break;
            }
            prev_max_diff = max_diff;
            //printf("%d %d %d\n", min_not_found, k_min, threads_per_level[k_min]);
            if (max_diff > 0.0){
               threads_per_level[k_min]--;
               threads_per_level[k_max]++; 
            }
            else {
               threads_per_level[k_min]++;
               threads_per_level[k_max]--;
            }
            //for (int k = 0; k < num_levels; k++){
            //   printf("\tlevel %d: %f, %f, %d\n",
            //          k,
            //          all_data->grid.frac_level_work[k],
            //          (double)threads_per_level[k]/(double)all_data->input.num_threads,
            //          threads_per_level[k]);
            //}
         }
         int t = 0;
         for (int k = 0; k < num_levels; k++){
            all_data->thread.barrier_root[k] = t;
            for (int tt = 0; tt < threads_per_level[k]; tt++){
               all_data->thread.thread_levels[t].push_back(k);
               all_data->thread.level_threads[k].push_back(t);
               all_data->thread.barrier_flags[k][t] = 0;
               if (t < num_threads-1){
                  t++;
               }
            }
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

void SMEM_CSRMatrixGetLoadBalancedPartitionBoundary(AllData *all_data,
                                                    hypre_CSRMatrix *A,
                                                    int size_per_thread,
                                                    int num_threads, int t,
                                                    int *ns, int *ne)
{
   int n, nnz;

   n = hypre_CSRMatrixNumRows(A);
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   if (t == 0){
      *ns = 0;
   }
   else {
      *ns = hypre_LowerBound(A_i, A_i + n, size_per_thread * t) - A_i;
   }

   if (t == num_threads-1){
      *ne = n;
   }
   else {
      *ne = hypre_LowerBound(A_i, A_i + n, size_per_thread * (t+1)) - A_i;
   }
}

void PartitionGrids(AllData *all_data)
{
   int num_level_threads;
   int num_levels =  all_data->grid.num_levels;
   int num_rows, num_cols;

   if (all_data->input.thread_part_type == ALL_LEVELS){
      all_data->thread.A_ns = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.A_ne = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.R_ns = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.R_ne = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.P_ns = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.P_ne = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.row_ns = (int **)malloc(num_levels * sizeof(int *));
      all_data->thread.row_ne = (int **)malloc(num_levels * sizeof(int *));
     
      for (int level = 0; level < num_levels; level++){
         all_data->thread.A_ns[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.A_ne[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.R_ns[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.R_ne[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.P_ns[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.P_ne[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.row_ns[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
         all_data->thread.row_ne[level] = (int *)malloc(all_data->input.num_threads * sizeof(int));
      }

      int finest_level;
      if (all_data->input.res_compute_type == GLOBAL){
         all_data->thread.A_ns_global = (int *)calloc(all_data->input.num_threads, sizeof(int));
         all_data->thread.A_ne_global = (int *)calloc(all_data->input.num_threads, sizeof(int));
	 int n = all_data->grid.n[0];
	 int size = n/all_data->input.num_threads;
	 int rest = n - size*all_data->input.num_threads;
	 for (int t = 0; t < all_data->input.num_threads; t++){
            if (t < rest){
               all_data->thread.A_ns_global[t] = t*size + t;
               all_data->thread.A_ne_global[t] = (t + 1)*size + t + 1;
            }
            else{
               all_data->thread.A_ns_global[t] = t*size + rest;
               all_data->thread.A_ne_global[t] = (t + 1)*size + rest;
            }
	 }
	 finest_level = 0;
      }
      else{
         finest_level = 0;
      }

      for (int level = finest_level; level < num_levels; level++){
         num_level_threads = all_data->thread.level_threads[level].size();
         for (int inner_level = 0; inner_level < num_levels; inner_level++){
            HYPRE_Int n, nnz, nnz_per_thread, n_per_thread;
            for (int i = 0; i < num_level_threads; i++){
               int t = all_data->thread.level_threads[level][i];
               int rest, size;
               int shift_t = t - all_data->thread.level_threads[level][0];
               
               hypre_CSRMatrix *A = all_data->matrix.A[inner_level];
               nnz = hypre_CSRMatrixNumNonzeros(A);
               n = hypre_CSRMatrixNumRows(A);
               nnz_per_thread = (nnz + num_level_threads - 1)/num_level_threads;
               SMEM_CSRMatrixGetLoadBalancedPartitionBoundary(all_data, A, nnz_per_thread, num_level_threads, shift_t, 
                                                              &(all_data->thread.A_ns[inner_level][t]), &(all_data->thread.A_ne[inner_level][t]));

               if (inner_level < num_levels-1){
                  hypre_CSRMatrix *P = all_data->matrix.P[inner_level];
                  nnz = hypre_CSRMatrixNumNonzeros(P);
                  nnz_per_thread = (nnz + num_level_threads - 1)/num_level_threads;
                  SMEM_CSRMatrixGetLoadBalancedPartitionBoundary(all_data, P, nnz_per_thread, num_level_threads, shift_t,
                                                                 &(all_data->thread.P_ns[inner_level][t]), &(all_data->thread.P_ne[inner_level][t]));
                  if (all_data->input.construct_R_flag == 1){
                     hypre_CSRMatrix *R = all_data->matrix.R[inner_level];
                     nnz = hypre_CSRMatrixNumNonzeros(R);
                     nnz_per_thread = (nnz + num_level_threads - 1)/num_level_threads;
                     SMEM_CSRMatrixGetLoadBalancedPartitionBoundary(all_data, R, nnz_per_thread, num_level_threads, shift_t,
                                                                    &(all_data->thread.R_ns[inner_level][t]), &(all_data->thread.R_ne[inner_level][t]));
                  }
                  else {
                     all_data->thread.R_ns[inner_level][t] = all_data->thread.P_ns[inner_level][t];
                     all_data->thread.R_ne[inner_level][t] = all_data->thread.P_ne[inner_level][t];
                  }
               }
            }

            int *parts = (int *)calloc(num_level_threads, sizeof(int));
            int *disps = (int *)calloc(num_level_threads+1, sizeof(int));
            int count = 0;
            hypre_CSRMatrix *A = all_data->matrix.A[inner_level];
            n = hypre_CSRMatrixNumRows(A);
            while (count < n){
               for (int i = 0; i < num_level_threads; i++){
                  parts[i]++;
                  count++;
                  if (count == n){
                     break;
                  }
               }
            }
            disps[0] = 0;
            for (int i = 0; i < num_level_threads; i++){
               disps[i+1] = disps[i] + parts[i]; 
            }
            for (int i = 0; i < num_level_threads; i++){
               int t = all_data->thread.level_threads[level][i];
               all_data->thread.row_ns[inner_level][t] = disps[i];
               all_data->thread.row_ne[inner_level][t] = disps[i+1];
            }
            free(parts);
            free(disps);
         }
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
            if (t < rest){
               all_data->thread.A_ns[level][t] = t*size + t;
               all_data->thread.A_ne[level][t] = (t + 1)*size + t + 1;
            }
            else {
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
   all_data->grid.level_work = (int *)calloc(all_data->grid.num_levels, sizeof(int));

   int finest_level;
   if (all_data->input.res_compute_type == GLOBAL){
      finest_level = 1;
   }
   else{
      finest_level = 0;
   }

   for (int level = finest_level; level < all_data->grid.num_levels; level++){
      coarsest_level = all_data->grid.num_levels-1;
      finest_level = 0;
      all_data->grid.level_work[level] = 0;
      if (all_data->input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
         //all_data->grid.level_work[coarsest_level] += hypre_CSRMatrixNumRows(all_data->matrix.A[coarsest_level]);
         for (int inner_level = coarsest_level-1; inner_level >= level; inner_level--){
            fine_grid = inner_level;
            coarse_grid = inner_level + 1;
            all_data->grid.level_work[level] += hypre_CSRMatrixNumNonzeros(all_data->matrix.P[inner_level]);
            //all_data->grid.level_work[level] += hypre_CSRMatrixNumRows(all_data->matrix.A[fine_grid]);
            //all_data->grid.level_work[level] += hypre_CSRMatrixNumRows(all_data->matrix.A[coarse_grid]);
         }
         for (int inner_level = finest_level; inner_level < level; inner_level++){
            fine_grid = inner_level;
            coarse_grid = inner_level + 1;
            all_data->grid.level_work[level] += hypre_CSRMatrixNumNonzeros(all_data->matrix.R[inner_level]);
            //all_data->grid.level_work[level] += hypre_CSRMatrixNumRows(all_data->matrix.A[fine_grid]);
            //all_data->grid.level_work[level] += hypre_CSRMatrixNumRows(all_data->matrix.A[coarse_grid]);
         }
         all_data->grid.level_work[level] += 2*hypre_CSRMatrixNumNonzeros(all_data->matrix.A[level]);
         //all_data->grid.level_work[level] += hypre_CSRMatrixNumRows(all_data->matrix.A[level]);
         //all_data->grid.level_work[level] += 6*hypre_CSRMatrixNumRows(all_data->matrix.A[level]);
      }
      else {
         if (all_data->input.res_compute_type == GLOBAL){
            all_data->grid.level_work[level] += 
               2*hypre_CSRMatrixNumNonzeros(all_data->matrix.A[0]) / all_data->grid.num_levels + 
                  hypre_CSRMatrixNumRows(all_data->matrix.A[0]);
         }
         else{
            all_data->grid.level_work[level] += 
               hypre_CSRMatrixNumNonzeros(all_data->matrix.A[0]) + hypre_CSRMatrixNumRows(all_data->matrix.A[0]);
         }

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
               all_data->grid.level_work[level] +=
                  hypre_CSRMatrixNumNonzeros(all_data->matrix.R[fine_grid]);

            }
            else if (all_data->input.solver == AFACX ||
                     all_data->input.solver == ASYNC_AFACX){
               if (level < all_data->grid.num_levels-1){
                  all_data->grid.level_work[level] +=
                     fine_grid * hypre_CSRMatrixNumNonzeros(all_data->matrix.R[fine_grid]);
               }
            }
         }

         fine_grid = level;
         coarse_grid = level + 1;
         if (level == all_data->grid.num_levels-1){
            all_data->grid.level_work[level] += 
               hypre_CSRMatrixNumNonzeros(all_data->matrix.A[fine_grid]);
         }
         else {
            if (all_data->input.solver == MULTADD ||
                all_data->input.solver == ASYNC_MULTADD){
               if (all_data->input.num_post_smooth_sweeps > 0 &&
                   all_data->input.num_pre_smooth_sweeps > 0){
                  all_data->grid.level_work[level] +=
                     all_data->input.num_fine_smooth_sweeps * 
           	     (hypre_CSRMatrixNumNonzeros(all_data->matrix.A[fine_grid]) +
           		hypre_CSRMatrixNumRows(all_data->matrix.A[fine_grid]));
               }
               else {
                  all_data->grid.level_work[level] += 
           	  hypre_CSRMatrixNumRows(all_data->matrix.A[fine_grid]);
               }
            }
            else if (all_data->input.solver == AFACX ||
                     all_data->input.solver == ASYNC_AFACX){
               all_data->grid.level_work[level] +=
                  (all_data->input.num_coarse_smooth_sweeps-1) * hypre_CSRMatrixNumNonzeros(all_data->matrix.A[coarse_grid]) +
                  hypre_CSRMatrixNumNonzeros(all_data->matrix.P[fine_grid]) +
                  hypre_CSRMatrixNumNonzeros(all_data->matrix.A[fine_grid]) +
                  (all_data->input.num_fine_smooth_sweeps-1) * hypre_CSRMatrixNumNonzeros(all_data->matrix.A[fine_grid]);
            }
         }

         coarsest_level = level;
         for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
            fine_grid = inner_level;
            coarse_grid = inner_level + 1;
            if (all_data->input.solver == MULTADD ||
                all_data->input.solver == ASYNC_MULTADD){
               all_data->grid.level_work[level] +=
                  hypre_CSRMatrixNumNonzeros(all_data->matrix.P[fine_grid]);

            }
            else if (all_data->input.solver == AFACX ||
                     all_data->input.solver == ASYNC_AFACX){
               all_data->grid.level_work[level] +=
                  hypre_CSRMatrixNumNonzeros(all_data->matrix.P[fine_grid]);
            }
         }
      }
   }
   all_data->grid.tot_work = 0;
   all_data->grid.frac_level_work = (double *)calloc(all_data->grid.num_levels, sizeof(double));
   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->grid.tot_work += all_data->grid.level_work[level];
   }
   for (int level = 0; level < all_data->grid.num_levels; level++){
      all_data->grid.frac_level_work[level] = (double)all_data->grid.level_work[level] / (double)all_data->grid.tot_work;
   }
}

void SmoothTransfer(AllData *all_data,
                    hypre_CSRMatrix *P,
                    hypre_CSRMatrix *R,
                    int level)
{
   if (all_data->input.num_post_smooth_sweeps == 0 &&
       all_data->input.num_pre_smooth_sweeps == 0){
      return;
   }
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

   vector<double> diag(num_rows);

  // if (level > 0){
      if (all_data->input.smooth_interp_type == JACOBI ||
          all_data->input.smooth_interp_type == HYBRID_JACOBI_GAUSS_SEIDEL){
         for (int i = 0; i < num_rows; i++){
            diag[i] = A_data[A_i[i]];
         }
         for (int i = 0; i < num_rows; i++){
            G_data[A_i[i]] = GT_data[A_i[i]] = 1.0 - all_data->input.smooth_weight;
            for (int jj = A_i[i]+1; jj < A_i[i+1]; jj++){
               G_data[jj] = -all_data->input.smooth_weight * A_data[jj] / diag[i];
               GT_data[jj] = -all_data->input.smooth_weight * A_data[jj] / diag[A_j[jj]];
            }
         }
      }
      else{
         for (int i = 0; i < num_rows; i++){
            G_data[A_i[i]] = GT_data[A_i[i]] = 1.0 - A_data[A_i[i]] / all_data->matrix.L1_row_norm[level][i];
            for (int jj = A_i[i]+1; jj < A_i[i+1]; jj++){
               G_data[jj] = -A_data[jj] / all_data->matrix.L1_row_norm[level][i];
               GT_data[jj] = -A_data[jj] / all_data->matrix.L1_row_norm[level][A_j[jj]];
            }
         }
      }
  // }
  // else{
  //    for (int i = 0; i < num_rows; i++){
  //       G_data[A_i[i]] = GT_data[A_i[i]] = 1.0;
  //       for (int jj = A_i[i]+1; jj < A_i[i+1]; jj++){
  //          G_data[jj] = 0.0;
  //          GT_data[jj] = 0.0;
  //       }
  //    }
  // }

   hypre_CSRMatrixI(G) = A_i;
   hypre_CSRMatrixJ(G) = A_j;
   hypre_CSRMatrixData(G) = G_data;
   hypre_CSRMatrixI(GT) = A_i;
   hypre_CSRMatrixJ(GT) = A_j;
   hypre_CSRMatrixData(GT) = GT_data;
   if (all_data->input.num_post_smooth_sweeps > 0){
      EigenMatMat(all_data, G, P, all_data->matrix.P[level]);
   }
   if (all_data->input.num_pre_smooth_sweeps > 0){
      EigenMatMat(all_data, RT, GT, all_data->matrix.R[level]);
   }
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
   eigen_C = (eigen_A * eigen_B);
   eigen_C.makeCompressed();

  // vector<double> row_sum(eigen_C.rows());
  // for (int i = 0; i < eigen_C.rows(); i++){
  //    row_sum[i] = 0;
  //    for (EigenSpMat::InnerIterator it(eigen_C,i); it; ++it) {
  //       row_sum[i] += it.value();
  //    }
  // }
  // eigen_C = (eigen_A * eigen_B).pruned(.5,1.0);
  // for (int i = 0; i < eigen_C.rows(); i++){
  //    double pruned_row_sum = 0;
  //    for (EigenSpMat::InnerIterator it(eigen_C,i); it; ++it) {
  //       pruned_row_sum += it.value();
  //    }
  //    for (EigenSpMat::InnerIterator it(eigen_C,i); it; ++it) {
  //       it.valueRef() /= pruned_row_sum / row_sum[i];
  //    }
  // }

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
   #pragma omp parallel for
   for (int i = 0; i < num_rows; i++){
      for (int jj = A_i[i], kk = j_vector[i].size()-1; jj < A_i[i+1]; jj++, kk--){
         A_j[jj] = j_vector[i][kk];
         A_data[jj] = data_vector[i][kk];
        // A_j[k] = j_vector[i].back();
        // A_data[k] = data_vector[i].back();
        // j_vector[i].pop_back();
        // data_vector[i].pop_back();
        // k++;
      }
   }

   #pragma omp parallel for
   for (int i = 0; i < num_rows; i++){
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         if (i == A_j[jj]){
            int j_tmp = A_j[A_i[i]];
            A_j[A_i[i]] = A_j[jj];
            A_j[jj] = j_tmp;

            double data_tmp = A_data[A_i[i]];
            A_data[A_i[i]] = A_data[jj];
            A_data[jj] = data_tmp;
            break;
         }
      }
   }

   hypre_CSRMatrixI(A) = A_i;
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_data;
}

void BuildExtendedMatrix(AllData *all_data,
                         hypre_CSRMatrix **A_array,
                         hypre_CSRMatrix **P_array,
                         hypre_CSRMatrix **R_array,
                         hypre_CSRMatrix **B_ptr)
{
   hypre_CSRMatrix *Q, *M, *AP, *RA, *R;
   hypre_CSRMatrix *B = NULL;
   double matmul_wtime = 0, csr_to_vec_wtime = 0, start;

   int num_levels = all_data->grid.num_levels;
   all_data->grid.disp = (int *)malloc((num_levels + 1) * sizeof(int));
   int NNZ, N = 0;
   all_data->grid.disp[0] = 0;
   for (int level = 0; level < num_levels; level++){
      HYPRE_Int num_rows;
      if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
         num_rows = hypre_CSRMatrixNumRows(A_array[level]);
      }
      else {
         num_rows = hypre_CSRMatrixNumRows(A_array[0]);
      }
      all_data->grid.disp[level+1] = all_data->grid.disp[level] + num_rows;
      N += num_rows;
   }

   vector<vector<HYPRE_Int>> col_vec(N, vector<HYPRE_Int>(0));
   vector<vector<HYPRE_Real>> val_vec(N, vector<HYPRE_Real>(0));
   /* loop over row blocks */
   for (int level = 0; level < num_levels; level++){
      int level_start = all_data->grid.disp[level];
      start = omp_get_wtime();
      CSR_to_Vector(all_data, A_array[level], &col_vec, &val_vec, level_start, level_start);
      csr_to_vec_wtime += omp_get_wtime() - start;
      /* loop over column blocks */
      if (all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
         for (int inner_level = level+1; inner_level < num_levels; inner_level++){
            int inner_level_start = all_data->grid.disp[inner_level];
            AP = A_array[level];
            hypre_CSRMatrixTranspose(A_array[level], &RA, 1);
            start = omp_get_wtime();
            for (int k = level; k < inner_level; k++){
               Q = hypre_CSRMatrixMultiply(AP, P_array[k]);
               AP = Q;
               M = hypre_CSRMatrixMultiply(R_array[k], RA);
               RA = M;
            }
            matmul_wtime += omp_get_wtime() - start;
            
            start = omp_get_wtime(); 
            CSR_to_Vector(all_data, AP, &col_vec, &val_vec, level_start, inner_level_start);
            CSR_to_Vector(all_data, RA, &col_vec, &val_vec, inner_level_start, level_start);
            csr_to_vec_wtime += omp_get_wtime() - start;
         }
      }
      else {
         for (int inner_level = num_levels-2; inner_level >= level; inner_level--){
            int inner_level_start = all_data->grid.disp[inner_level+1];
            start = omp_get_wtime();
            CSR_to_Vector(all_data, P_array[inner_level], &col_vec, &val_vec, level_start, inner_level_start);
            //R = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
            //hypre_CSRMatrixTranspose(P_array[inner_level], &R, 1);
            //CSR_to_Vector(all_data, R, &col_vec, &val_vec, inner_level_start, level_start);
            csr_to_vec_wtime += omp_get_wtime() - start;
         }
         for (int inner_level = 0; inner_level < level; inner_level++){
            int inner_level_start = all_data->grid.disp[inner_level];
            if (all_data->input.construct_R_flag == 0){
               R = (hypre_CSRMatrix *)malloc(sizeof(hypre_CSRMatrix));
               hypre_CSRMatrixTranspose(R_array[inner_level], &R, 1);
            }
            else {
               R = R_array[inner_level];
            }
            start = omp_get_wtime();
            CSR_to_Vector(all_data, R, &col_vec, &val_vec, level_start, inner_level_start);
            csr_to_vec_wtime += omp_get_wtime() - start;

            if (all_data->input.construct_R_flag == 0) {
               free(R);
            }
         }
      }
   }

   NNZ = 0;
   for (int i = 0; i < N; i++){
      NNZ += col_vec[i].size();
   }
   B = hypre_CSRMatrixCreate(N, N, NNZ);
   start = omp_get_wtime();
   StdVector_to_CSR(B, col_vec, val_vec);
   csr_to_vec_wtime += omp_get_wtime() - start;
   if (all_data->input.format_output_flag == 0){
      printf("matmul wtime %e, csr_to_vec wtime %e\n", matmul_wtime, csr_to_vec_wtime);
   }
   *B_ptr = B;

 //  HYPRE_IJMatrix Bij;
 //  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, N-1, 0, N-1, &Bij);
 //  HYPRE_IJMatrixSetObjectType(Bij, HYPRE_PARCSR);
 //  HYPRE_IJMatrixInitialize(Bij);
 //  for (int i = 0; i < N; i++){
 //     int nnz = col_vec[i].size();
 //     double *values = (double *)malloc(nnz * sizeof(double));
 //     int *cols = (int *)malloc(nnz * sizeof(int));

 //     for (int j = 0; j < nnz; j++){
 //        cols[j] = col_vec[i][j];
 //        values[j] = val_vec[i][j];
 //     }

 //     HYPRE_IJMatrixSetValues(Bij, 1, &nnz, &i, cols, values);

 //     free(values);
 //     free(cols);
 //  }
 //  HYPRE_IJMatrixAssemble(Bij);
 //  void *object;
 //  HYPRE_IJMatrixGetObject(Bij, &object);
 // // *B_ptr = (hypre_ParCSRMatrix *)object;
 //  *B_ptr = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)object);
}

void CSR_to_Vector(AllData *all_data,
                   hypre_CSRMatrix *A,
                   vector<vector<HYPRE_Int>> *col_vec,
                   vector<vector<HYPRE_Real>> *val_vec,
                   int row_start, int col_start)
{
   HYPRE_Real *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   #pragma omp parallel for
   for (int i = 0; i < num_rows; i++){
      for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
         HYPRE_Int ii = A_j[jj];
         int row = row_start+i;
         int col = col_start+ii;
         double val = A_data[jj];
        
         //printf("%d %d\n", row, (*col_vec).size()); 
        // if (row_start == col_start && row == col){
            (*col_vec)[row].push_back(col);
            (*val_vec)[row].push_back(val);
        // }
        // else { 
        //    (*col_vec)[row].insert((*col_vec)[row].begin(), col);
        //    (*val_vec)[row].insert((*val_vec)[row].begin(), val);
        // }
      }
   }
}

void SMEM_BuildMatrix(AllData *all_data)
{
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_IJVector x;
   HYPRE_ParVector rhs;

   //if (all_data->input.test_problem == LAPLACE_3D27PT){
   //   Laplacian_3D_27pt(&(all_data->hypre.parcsr_A),
   //                     all_data->matrix.nx,
   //                     all_data->matrix.ny,
   //                     all_data->matrix.nz);
   //}
   //else if (all_data->input.test_problem == LAPLACE_3D7PT){
   //   Laplacian_3D_7pt(&(all_data->hypre.parcsr_A),
   //                    all_data->matrix.nx,
   //                    all_data->matrix.ny,
   //                    all_data->matrix.nz);
   //}
   if (all_data->input.test_problem == LAPLACE_2D5PT){
      Laplacian_2D_5pt(&A, all_data->matrix.n);
      HYPRE_IJMatrixAssemble(A);
      HYPRE_IJMatrixGetObject(A, (void**) &(all_data->hypre.parcsr_A));
   }
   else if ((all_data->input.test_problem == MFEM_LAPLACE) ||
            (all_data->input.test_problem == MFEM_LAPLACE_AMR)){
#ifdef USE_MFEM
      BuildMfemMatrix(all_data,
                      &(all_data->hypre.parcsr_A),
                      &(all_data->hypre.par_b),
                      MPI_COMM_WORLD);
#else
      printf("ERROR: must compile with mfem to use mfem test problems\n");
      MPI_Finalize();
      exit(1); 
#endif
      //HYPRE_IJMatrixAssemble(A);
      //HYPRE_IJMatrixGetObject(A, (void**) &(all_data->hypre.parcsr_A));
   }
   else if ((all_data->input.test_problem == MFEM_ELAST) ||
            (all_data->input.test_problem == MFEM_ELAST_AMR)){
#ifdef USE_MFEM
      BuildMfemMatrix(all_data,
                      &(all_data->hypre.parcsr_A),
                      &(all_data->hypre.par_b),
                      MPI_COMM_WORLD);
#else
      printf("ERROR: must compile with mfem to use mfem test problems\n");
#endif
      //HYPRE_IJMatrixAssemble(A);
      //HYPRE_IJMatrixGetObject(A, (void**) &(all_data->hypre.parcsr_A));
   }
   else if (all_data->input.test_problem == MFEM_MAXWELL){
#ifdef USE_MFEM
      BuildMfemMatrix(all_data,
                      &(all_data->hypre.parcsr_A),
                      &(all_data->hypre.par_b),
                      MPI_COMM_WORLD);
#else
      printf("ERROR: must compile with mfem to use mfem test problems\n");
#endif
      //HYPRE_IJMatrixAssemble(A);
      //HYPRE_IJMatrixGetObject(A, (void**) &(all_data->hypre.parcsr_A));
   }
   else if (all_data->input.test_problem == MATRIX_FROM_FILE){
      FILE *mat_file = fopen(all_data->input.mat_file_str, "rb");
      if (mat_file != NULL){
         ReadBinary_fread_HypreParCSR(mat_file, &(all_data->hypre.parcsr_A), 1, 0);
         fclose(mat_file);
      }
      else {
         printf("ERROR: incorrect matrix file \"%s\"\n", all_data->input.mat_file_str);
         MPI_Finalize();
         exit(1);
      }
   }
   else {
      BuildHypreMatrix(all_data,
                       &(all_data->hypre.parcsr_A),
                       &rhs,
                       MPI_COMM_WORLD,
                       all_data->matrix.nx, all_data->matrix.ny, all_data->matrix.nz,
                       all_data->matrix.difconv_cx, all_data->matrix.difconv_cy, all_data->matrix.difconv_cz,
                       all_data->matrix.difconv_ax, all_data->matrix.difconv_ay, all_data->matrix.difconv_az,
                       all_data->matrix.vardifconv_eps,
                       all_data->matrix.difconv_atype);
   }

   //char buffer[100] = "A.txt";
   //PrintCSRMatrix(hypre_ParCSRMatrixDiag(all_data->hypre.parcsr_A), buffer, 0);
}

void SMEM_SetHypreParameters(AllData *all_data)
{
   HYPRE_BoomerAMGCreate(&(all_data->hypre.solver));

   HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, all_data->hypre.print_level);
   HYPRE_BoomerAMGSetMaxRowSum(all_data->hypre.solver, 1.0);
   HYPRE_BoomerAMGSetPostInterpType(all_data->hypre.solver, 0);
   HYPRE_BoomerAMGSetInterpType(all_data->hypre.solver, all_data->hypre.interp_type);
   HYPRE_BoomerAMGSetRestriction(all_data->hypre.solver, 0);
   HYPRE_BoomerAMGSetCoarsenType(all_data->hypre.solver, all_data->hypre.coarsen_type);
   HYPRE_BoomerAMGSetMaxLevels(all_data->hypre.solver, all_data->hypre.max_levels);
   HYPRE_BoomerAMGSetAggNumLevels(all_data->hypre.solver, all_data->hypre.agg_num_levels);
   HYPRE_BoomerAMGSetStrongThreshold(all_data->hypre.solver, all_data->hypre.strong_threshold);
   HYPRE_BoomerAMGSetPMaxElmts(all_data->hypre.solver, all_data->hypre.P_max_elmts);
   HYPRE_BoomerAMGSetNumFunctions(all_data->hypre.solver, all_data->hypre.num_functions);
   HYPRE_BoomerAMGSetMeasureType(all_data->hypre.solver, 1);
   HYPRE_BoomerAMGSetRelaxType(all_data->hypre.solver, all_data->hypre.relax_type);
   HYPRE_BoomerAMGSetAddRelaxType(all_data->hypre.solver, all_data->hypre.relax_type);
   HYPRE_BoomerAMGSetRelaxWt(all_data->hypre.solver, all_data->input.smooth_weight);
   HYPRE_BoomerAMGSetAddRelaxWt(all_data->hypre.solver, all_data->input.smooth_weight);
   HYPRE_BoomerAMGSetCycleRelaxType(all_data->hypre.solver, all_data->hypre.relax_type, 3);
   HYPRE_BoomerAMGSetNumSweeps(all_data->hypre.solver, all_data->input.num_pre_smooth_sweeps);
   //HYPRE_BoomerAMGSetCycleRelaxType(all_data->hypre.solver, 99, 3);

   if (all_data->input.solver == BPX ||
       all_data->input.solver == PAR_BPX ||
       all_data->input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX ||
       all_data->input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX){
      HYPRE_BoomerAMGSetAdditive(all_data->hypre.solver, 0);
      HYPRE_BoomerAMGSetSimple(all_data->hypre.solver, all_data->input.simple_jacobi_flag);
   }
   //HYPRE_BoomerAMGSetAdditive(all_data->hypre.solver, 0);

   int num_rows = hypre_ParCSRMatrixNumRows(all_data->hypre.parcsr_A);

   HYPRE_IJVector b;
   HYPRE_IJVector x;

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, num_rows-1, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, num_rows-1, &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);

   int *rows;
   rows = (int*)calloc(num_rows, sizeof(int));
   for (int i = 0; i < num_rows; i++){
      rows[i] = i;
   }

   double *rhs_values, *x_values;
   rhs_values = (double*) calloc(num_rows, sizeof(double));
   x_values = (double*) calloc(num_rows, sizeof(double));

   srand(0);

   //if ((all_data->input.test_problem == MFEM_ELAST) ||
   //    (all_data->input.test_problem == MFEM_ELAST_AMR) ||
   //    (all_data->input.test_problem == MFEM_LAPLACE) ||
   //    (all_data->input.test_problem == MFEM_LAPLACE_AMR)){
   //   //for (int i = 0; i < num_rows; i++){
   //   //   rhs_values[i] = all_data->hypre.b_values[i];
   //   //}
   //}
   //else {
      for (int i = 0; i < num_rows; i++){
         rhs_values[i] = RandDouble(-1.0, 1.0);
      }
      HYPRE_IJVectorSetValues(b, num_rows, rows, rhs_values);
      HYPRE_IJVectorAssemble(b);
      HYPRE_IJVectorGetObject(b, (void **)&(all_data->hypre.par_b));
   //}

   for (int i = 0; i < num_rows; i++){
      x_values[i] = 0.0;
   }

   HYPRE_IJVectorSetValues(x, num_rows, rows, x_values);
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **)&(all_data->hypre.par_x));

   free(x_values);
   free(rhs_values);
   free(rows);
}
