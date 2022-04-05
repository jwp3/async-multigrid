#include "Main.hpp"
#include "Misc.hpp"
#include "SMEM_MatVec.hpp"
#include "lobpcg.h"
#include "SMEM_Sync_AMG.hpp"

#include <slepceps.h>

using namespace std;

void EigsPower(AllData *all_data);
void BPXCycle(AllData *all_data);

void EigsHypreLOBPCG(AllData *all_data);
void SetupMGPrecondMatvec(HYPRE_MatvecFunctions * mv);
void *MGPrecondMatvecCreate(void *A, void *x);
HYPRE_Int MGPrecondMatvec(void *matvec_data,
                          HYPRE_Complex  alpha,
                          void *A,
                          void *x,
                          HYPRE_Complex beta,
                          void *y);
HYPRE_Int MGPrecondMatvecDestroy(void *matvec_data);

PetscErrorCode Slepc_MatMult(Mat A, Vec x, Vec y);
void EigsSlepc(AllData *all_data);

void ChebySetup(AllData *all_data)
{
   double start = omp_get_wtime();
   if (all_data->input.cheby_eig_type == CHEBY_EIG_HYPRE_LOBPCG){
      EigsHypreLOBPCG(all_data);
   }
   else if (all_data->input.cheby_eig_type == CHEBY_EIG_SLEPC){
      EigsSlepc(all_data);
   }
   else {
      EigsPower(all_data);
   }
   double eig_time = omp_get_wtime() - start;

   double alpha = all_data->cheby.alpha;
   double beta = all_data->cheby.beta;
   int num_threads = all_data->input.num_threads;
   if (all_data->input.format_output_flag == 0)
      printf("CHEBY: eig max %e, eig min %e, k_2 %.2f, time %e\n",
             beta,  alpha, beta/alpha, eig_time);
   all_data->cheby.mu = (beta + alpha) / (beta - alpha);
   all_data->cheby.delta = 2.0 / (beta + alpha);

   double omega = 2.0 / (1.0 + sqrt(1.0 - pow(all_data->cheby.mu, -2.0)));
   all_data->cheby.c_prev = (double *)malloc(num_threads * sizeof(double));
   all_data->cheby.c = (double *)malloc(num_threads * sizeof(double));
   all_data->cheby.omega = (double *)malloc(num_threads * sizeof(double));
   for (int t = 0; t < num_threads; t++){
      all_data->cheby.c_prev[t] = 1.0;
      all_data->cheby.c[t] = all_data->cheby.mu;
      all_data->cheby.omega[t] = omega;
   }
}

void EigsSlepc(AllData *all_data)
{
   Mat A;
   EPS eps;
   EPSType type;
   PetscInt nconv, N;
   PetscScalar eigr, eigi;
   double eig_max, eig_min;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);

   N = num_rows;
   SlepcInitialize(NULL, NULL, NULL, NULL);
   MatCreateShell(PETSC_COMM_WORLD, N, N, N, N, all_data, &A);
   MatShellSetOperation(A, MATOP_MULT, (void(*)(void))Slepc_MatMult);

   EPSCreate(PETSC_COMM_WORLD, &eps);
   EPSSetOperators(eps, A, NULL);
   EPSSetProblemType(eps, EPS_NHEP);
   EPSSetTolerances(eps, all_data->input.cheby_eig_tol, all_data->input.cheby_eig_max_iters);
   EPSSetType(eps, EPSARNOLDI);
   EPSSetFromOptions(eps);

   EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
   EPSSolve(eps);
   EPSGetConverged(eps, &nconv);
   eig_max = -1;
   if (all_data->input.format_output_flag == 0) printf("\n");
   for (int i = 0; i < nconv; i++) {
      EPSGetEigenpair(eps, i, &eigr, &eigi, NULL, NULL);
      if (eigr > eig_max) eig_max = eigr;
      if (all_data->input.format_output_flag == 0) printf("SLEPC LM eig %e\n", eigr);
   }
   if (all_data->input.format_output_flag == 0) printf("\n");

   EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
   EPSSolve(eps);
   EPSGetConverged(eps, &nconv);
   eig_min = DBL_MAX;
   for (int i = 0; i < nconv; i++) {
      EPSGetEigenpair(eps, i, &eigr, &eigi, NULL, NULL);
      if (eigr < eig_min) eig_min = eigr;
      if (all_data->input.format_output_flag == 0) printf("SLEPC SM eig %e\n", eigr);
   }
   if (all_data->input.format_output_flag == 0) printf("\n");
   if (all_data->input.format_output_flag == 0){
      int its;
      EPSGetIterationNumber(eps, &its);
      printf("SLEPC iterations = %d\n", its);
   }
   if (all_data->input.format_output_flag == 0) printf("\n");

   all_data->cheby.beta = eig_max;
   all_data->cheby.alpha = eig_min;

   EPSDestroy(&eps);
   MatDestroy(&A);
   SlepcFinalize();
}

PetscErrorCode Slepc_MatMult(Mat A, Vec x, Vec y)
{
   void *ctx;
   const PetscScalar *px;
   PetscScalar *py;

   MatShellGetContext(A, &ctx);
   VecGetArrayRead(x, &px);
   VecGetArray(y, &py);

   AllData *all_data = (AllData *)ctx;

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector *u = U_array[0];
   hypre_ParVector *f = F_array[0];
   HYPRE_Real *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_array[0]);

   for (int i = 0; i < num_rows; i++){
      all_data->vector.y[0][i] = f_data[i];
      u_data[i] = px[i];
   }

   hypre_ParCSRMatrixMatvec(1.0,
                            A_array[0],
                            u,
                            0.0,
                            f);

   if (all_data->input.solver == MULT){
      for (int i = 0; i < num_rows; i++){
         hypre_VectorData(hypre_ParVectorLocalVector(all_data->hypre.par_b))[i] = f_data[i];
         hypre_VectorData(hypre_ParVectorLocalVector(all_data->hypre.par_x))[i] = 0.0;
      }
      HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, 0);
      HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, 1);
      HYPRE_BoomerAMGSetNumSweeps(all_data->hypre.solver, 1);
      HYPRE_BoomerAMGSolve(all_data->hypre.solver,
                           all_data->hypre.parcsr_A,
                           all_data->hypre.par_b,
                           all_data->hypre.par_x);
      for (int i = 0; i < num_rows; i++){
         py[i] = hypre_VectorData(hypre_ParVectorLocalVector(all_data->hypre.par_x))[i];
      }

      //for (int i = 0; i < all_data->grid.n[0]; i++){
      //   all_data->vector.r[0][i] = f_data[i];
      //   all_data->vector.u[0][i] = 0.0;
      //}
      //#pragma omp parallel
      //{      
      //   SMEM_Sync_Parfor_Vcycle(all_data);
      //}
      //for (int i = 0; i < all_data->grid.n[0]; i++){
      //   py[i] = all_data->vector.u[0][i];
      //}
   }
   else {
      hypre_ParVectorSetConstantValues(u, 0.0);
      BPXCycle(all_data);
      for (int i = 0; i < num_rows; i++){
         py[i] = u_data[i];
      }
   }

   for (int i = 0; i < num_rows; i++){
      f_data[i] = all_data->vector.y[0][i];
   }

   VecRestoreArrayRead(x, &px);
   VecRestoreArray(y, &py);
   return 0;
}

typedef struct
{
   HYPRE_Int (*Precond)(void*,void*,void*,void*);
   HYPRE_Int (*PrecondSetup)(void*,void*,void*,void*);

} hypre_LOBPCGPrecond;

typedef struct
{
   lobpcg_Tolerance              tolerance;
   HYPRE_Int                           maxIterations;
   HYPRE_Int                           verbosityLevel;
   HYPRE_Int                           precondUsageMode;
   HYPRE_Int                           iterationNumber;
   utilities_FortranMatrix*      eigenvaluesHistory;
   utilities_FortranMatrix*      residualNorms;
   utilities_FortranMatrix*      residualNormsHistory;

} lobpcg_Data;

#define lobpcg_tolerance(data)            ((data).tolerance)
#define lobpcg_absoluteTolerance(data)    ((data).tolerance.absolute)
#define lobpcg_relativeTolerance(data)    ((data).tolerance.relative)
#define lobpcg_maxIterations(data)        ((data).maxIterations)
#define lobpcg_verbosityLevel(data)       ((data).verbosityLevel)
#define lobpcg_precondUsageMode(data)     ((data).precondUsageMode)
#define lobpcg_iterationNumber(data)      ((data).iterationNumber)
#define lobpcg_eigenvaluesHistory(data)   ((data).eigenvaluesHistory)
#define lobpcg_residualNorms(data)        ((data).residualNorms)
#define lobpcg_residualNormsHistory(data) ((data).residualNormsHistory)

typedef struct
{

   lobpcg_Data                   lobpcgData;

   mv_InterfaceInterpreter*      interpreter;

   void*                         A;
   void*                         matvecData;
   void*                         precondData;

   void*                         B;
   void*                         matvecDataB;
   void*                         T;
   void*                         matvecDataT;

   hypre_LOBPCGPrecond           precondFunctions;

   HYPRE_MatvecFunctions*        matvecFunctions;

} hypre_LOBPCGData;

void SetupMGPrecondMatvec(HYPRE_MatvecFunctions * mv)
{
  mv->MatvecCreate = MGPrecondMatvecCreate;
  mv->Matvec = MGPrecondMatvec;
  mv->MatvecDestroy = MGPrecondMatvecDestroy;

  mv->MatMultiVecCreate = NULL;
  mv->MatMultiVec = NULL;
  mv->MatMultiVecDestroy = NULL;
}

void *MGPrecondMatvecCreate(void *A, void *x)
{
   void *matvec_data;
   matvec_data = NULL;
   return (matvec_data);
}

HYPRE_Int MGPrecondMatvec(void *matvec_data,
                          HYPRE_Complex  alpha,
                          void *A,
                          void *x,
                          HYPRE_Complex beta,
                          void *y)
{
   HYPRE_ParVector x_vec = (HYPRE_ParVector)x;
   HYPRE_ParVector y_vec = (HYPRE_ParVector)y;
   HYPRE_ParCSRMatrix A_mat = (HYPRE_ParCSRMatrix)A;

   AllData *all_data = (AllData *)matvec_data;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector *u = U_array[0];
   hypre_ParVector *f = F_array[0];
   HYPRE_Real *y_data = hypre_VectorData(hypre_ParVectorLocalVector(y_vec));
   HYPRE_Real *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));

   hypre_ParCSRMatrixMatvec(1.0,
                            A_mat,
                            x_vec,
                            0.0,
                            f);
   if (all_data->input.solver == MULT){
      HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, 0);
      HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, 1);
      HYPRE_BoomerAMGSolve(all_data->hypre.solver, A_array[0], f, u);
   }
   else {
      hypre_ParVectorSetConstantValues(u, 0.0);
      BPXCycle(all_data);
   }

   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A_mat);
   for (int i = 0; i < num_rows; i++){
      y_data[i] = all_data->input.cheby_eig_sign * alpha*u_data[i] + beta*y_data[i];
   }

   //hypre_ParCSRMatrixMatvec(alpha,
   //                         A_mat,
   //                         x_vec,
   //                         beta,
   //                         y_vec);
   return 0;
}

HYPRE_Int MGPrecondMatvecDestroy(void *matvec_data)
{
   return 0;
}

void EigsHypreLOBPCG(AllData *all_data)
{
   HYPRE_ParVector par_b;
   HYPRE_ParVector par_x;

   int num_rows = hypre_ParCSRMatrixNumRows(all_data->hypre.parcsr_A);

   HYPRE_IJVector b;
   HYPRE_IJVector x;

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, num_rows-1, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, num_rows-1, &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);


   double *rhs_values, *x_values;
   int *rows;

   rhs_values = (double*) calloc(num_rows, sizeof(double));
   x_values = (double*) calloc(num_rows, sizeof(double));
   rows = (int*)calloc(num_rows, sizeof(int));

   for (int i = 0; i < num_rows; i++){
      rows[i] = i;
   }

   HYPRE_IJVectorSetValues(b, num_rows, rows, rhs_values);
   HYPRE_IJVectorSetValues(x, num_rows, rows, x_values);

   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject(b, (void **)&(par_b));
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **)&(par_x));

   mv_InterfaceInterpreter* interpreter;
   HYPRE_MatvecFunctions matvec_fn;
   HYPRE_Solver lobpcg_solver;
   hypre_LOBPCGData *lobpcg_data;
   HYPRE_Int blockSize = 1;
   HYPRE_Real* eigenvalues;
   mv_MultiVectorPtr eigenvectors;
   int pcgMode = 1;
   int lobpcgSeed = 775;

   interpreter = (mv_InterfaceInterpreter *)malloc(sizeof(mv_InterfaceInterpreter));
   HYPRE_ParCSRSetupInterpreter(interpreter);
   //HYPRE_ParCSRSetupMatvec(&matvec_fn);
   SetupMGPrecondMatvec(&matvec_fn);

   eigenvectors = mv_MultiVectorCreateFromSampleVector(interpreter, blockSize, par_x);
   eigenvalues = (HYPRE_Real *)calloc(blockSize, sizeof(HYPRE_Real));

   HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &lobpcg_solver);
   HYPRE_LOBPCGSetMaxIter(lobpcg_solver, all_data->input.cheby_eig_max_iters);
   HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
   HYPRE_LOBPCGSetTol(lobpcg_solver, all_data->input.cheby_eig_tol);
   HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, all_data->hypre.print_level);

   HYPRE_LOBPCGSetup(lobpcg_solver, (HYPRE_Matrix)(all_data->hypre.parcsr_A), (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
   lobpcg_data = (hypre_LOBPCGData*)lobpcg_solver;
   lobpcg_data->matvecData = all_data;
  
   all_data->input.cheby_eig_sign = 1.0; 
   mv_MultiVectorSetRandom (eigenvectors, lobpcgSeed);
   HYPRE_LOBPCGSolve(lobpcg_solver, NULL, eigenvectors, eigenvalues);

   printf("HYPRE LobPCG %e\n", eigenvalues[0]);
 
   free(x_values);
   free(rhs_values);
   free(rows);
   free(interpreter); 
   free(eigenvalues); 
   HYPRE_LOBPCGDestroy(lobpcg_solver); 
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
}

void EigsPower(AllData *all_data)
{
   if (all_data->input.cheby_eig_max_iters <= 0) return;
   int eig_power_iters;
   double eig_max, eig_min;
   HYPRE_Real u_norm, y_norm;
   int num_threads = all_data->input.num_threads;

   srand(0);
  // srand(omp_get_wtime());

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);

   hypre_ParCSRMatrix *A = A_array[0];
   hypre_ParVector *u = U_array[0];
   hypre_ParVector *f = F_array[0];
   hypre_ParVector *v = hypre_ParAMGDataVtemp(amg_data);
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);

   HYPRE_Real *e = all_data->vector.e[0];
   HYPRE_Real *y = all_data->vector.y[0];

   HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Real *v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   //HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, all_data->input.eig_power_MG_max_iters);
   HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, 1);
   HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, 0);

   for (int i = 0; i < num_rows; i++){ 
      u_local_data[i] = 1.0;//RandDouble(-1.0, 1.0);
      y[i] = f_local_data[i];
   }
   eig_power_iters = 0;
   while (1){
      u_norm = sqrt(hypre_ParVectorInnerProd(u, u));
      hypre_ParVectorScale(1.0/u_norm, u);
      //hypre_ParVectorCopy(u, v);
      #pragma omp parallel for
      for (int i = 0; i < num_rows; i++){
         e[i] = u_local_data[i];
      }
      hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, f);
      hypre_ParVectorSetConstantValues(u, 0.0);
      if (all_data->input.solver == MULT){
         HYPRE_BoomerAMGSolve(all_data->hypre.solver, A, f, u);
      }
      else {
         BPXCycle(all_data);
      }

      eig_power_iters++;
      if (eig_power_iters == all_data->input.cheby_eig_max_iters) break;
   }
   #pragma omp parallel for
   for (int i = 0; i < num_rows; i++){
      v_local_data[i] = e[i];
   }
   eig_max = hypre_ParVectorInnerProd(v, u);

   for (int i = 0; i < num_rows; i++){
      u_local_data[i] = 1.0;//RandDouble(-1.0, 1.0);
   }
   eig_power_iters = 0;
   while (1){
      u_norm = sqrt(hypre_ParVectorInnerProd(u, u));
      hypre_ParVectorScale(1.0/u_norm, u);
      //hypre_ParVectorCopy(u, v);
      #pragma omp parallel for
      for (int i = 0; i < num_rows; i++){
         e[i] = u_local_data[i];
      }
      hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, f);
      hypre_ParVectorSetConstantValues(u, 0.0);
      if (all_data->input.solver == MULT){
         HYPRE_BoomerAMGSolve(all_data->hypre.solver, A, f, u);
      }
      else {
         BPXCycle(all_data);
      }

      #pragma omp parallel for
      for (int i = 0; i < num_rows; i++){
         v_local_data[i] = e[i];
      }
      eig_power_iters++;
      if (eig_power_iters == all_data->input.cheby_eig_max_iters) break;

      
      hypre_ParVectorAxpy(-eig_max, v, u);
   }
   eig_min = hypre_ParVectorInnerProd(v, u);

   all_data->cheby.beta = eig_max;// - all_data->input.b_eig_shift;
   all_data->cheby.alpha = eig_min;// + all_data->input.a_eig_shift;

   HYPRE_BoomerAMGSetMaxIter(all_data->hypre.solver, all_data->input.num_cycles);
   HYPRE_BoomerAMGSetPrintLevel(all_data->hypre.solver, all_data->hypre.print_level);

   srand(0);
   for (int i = 0; i < num_rows; i++){
      u_local_data[i] = 0.0;
      f_local_data[i] = all_data->vector.y[0][i];
   }  
}

void BPXCycle(AllData *all_data)
{
   int num_rows;
   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataPArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataRArray(amg_data);
   hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
   hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
   hypre_ParVector *v = hypre_ParAMGDataVtemp(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int relax_type = hypre_ParAMGDataGridRelaxType(amg_data)[1]; 
   //hypre_Vector **l1_norms_vec = hypre_ParAMGDataL1Norms(amg_data);


   HYPRE_Real *f_local_data, *u_local_data, *v_local_data, *A_data;
   HYPRE_Int *A_i, *A_j;

   v_local_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   for (int level = 0; level < num_levels-1; level++){   
      int fine_grid = level;
      int coarse_grid = level + 1;

      hypre_ParCSRMatrixMatvecT(1.0,
                                R_array[fine_grid],
                                F_array[fine_grid],
                                0.0,
                                F_array[coarse_grid]);
   }

   for (int level = 0; level < num_levels; level++){
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A_array[level]);
      A_data = hypre_CSRMatrixData(A_diag);
      A_i = hypre_CSRMatrixI(A_diag);
      A_j = hypre_CSRMatrixJ(A_diag);
      u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[level]));
      f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[level]));
      num_rows = hypre_CSRMatrixNumRows(A_diag);

      //HYPRE_Real relax_weight = hypre_ParAMGDataRelaxWeight(amg_data)[level];
      //HYPRE_Real *l1_norms_level = hypre_VectorData(l1_norms_vec[level])

      double *diag_scale;
      if (all_data->input.smoother == L1_JACOBI ||
          all_data->input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL){
         diag_scale = hypre_ParAMGDataL1Norms(amg_data)[level];
         //diag_scale = all_data->matrix.L1_row_norm[level];
      }
      else {
         diag_scale = all_data->matrix.A_diag[level];
      }

      #pragma omp parallel
      {
         int tid = omp_get_thread_num();

         HYPRE_Int ns = hypre_CSRMatrixGetLoadBalancedPartitionBegin(A_diag);
         HYPRE_Int ne = hypre_CSRMatrixGetLoadBalancedPartitionEnd(A_diag);
         int num_sweeps = all_data->input.num_pre_smooth_sweeps;
         if (all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL ||
             all_data->input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL){
            #pragma omp for
            for (int i = 0; i < num_rows; i++){
               v_local_data[i] = 0;
               u_local_data[i] = 0;
            }
            #pragma omp barrier
            for (int s = 0; s < num_sweeps; s++){
               for (int i = ns; i < ne; i++){
                  double res = f_local_data[i];
                  for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
                     int ii = A_j[jj];
                     if (ii >= ns && ii < ne){
                        res -= A_data[jj] * u_local_data[ii];
                     }
                     else {
                        res -= A_data[jj] * v_local_data[ii];
                     }
                  }
                  u_local_data[i] += res / diag_scale[i];
               }
               #pragma omp barrier
               for (int i = ns; i < ne; i++){
                  v_local_data[i] = u_local_data[i];
               }
               #pragma omp barrier
            }
         }
         else {
            #pragma omp for
            for (int i = 0; i < num_rows; i++){
               v_local_data[i] = 0;
               u_local_data[i] = 0;
            }
            #pragma omp barrier
            for (int s = 0; s < num_sweeps; s++){
               for (int i = ns; i < ne; i++){
                  double res = f_local_data[i];
                  for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
                     int ii = A_j[jj];
                     res -= A_data[jj] * v_local_data[ii];
                  }
                  u_local_data[i] +=  res / diag_scale[i];
               }
               #pragma omp barrier
               for (int i = ns; i < ne; i++){
                  v_local_data[i] = u_local_data[i];
               }
               #pragma omp barrier
            }
         }
      }
   }

   for (int level = num_levels-2; level >= 0; level--){
      int fine_grid = level;
      int coarse_grid = level + 1;

      hypre_ParCSRMatrixMatvec(1.0,
                               P_array[fine_grid],
                               U_array[coarse_grid],
                               1.0,
                               U_array[fine_grid]);
   }
}
