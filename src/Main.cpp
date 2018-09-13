/*
   Example 5

   Interface:    Linear-Algebraic (IJ)

   Compile with: make ex5

   Sample run:   mpirun -np 4 ex5

   Description:  This example solves the 2-D Laplacian problem with zero boundary
                 conditions on an n x n grid.  The number of unknowns is N=n^2.
                 The standard 5-point stencil is used, and we solve for the
                 interior nodes only.

                 This example solves the same problem as Example 3.  Available
                 solvers are AMG, PCG, and PCG with AMG or Parasails
                 preconditioners.  */

#include "Main.hpp"
#include "Laplacian.hpp"
#include "SEQ_MatVec.hpp"
#include "SEQ_AMG.hpp"
#include "SMEM_Setup.hpp"
#include "SMEM_Solve.hpp"
#include "Misc.hpp"


int main (int argc, char *argv[])
{
   int myid, num_procs;
   int N, n;

   int ilower, iupper;
   int local_size, extra;

   double start;
   int num_runs = 1;

   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   HYPRE_Solver solver;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   n = 33;
   /* Hypre parameters */
   int max_levels = 20;
   int solver_id = 0;
   HYPRE_Int agg_num_levels = 0;
   HYPRE_Int coarsen_type = 10;
   int hypre_print_level = 0;
   int hypre_solve_flag = 0;

   AllData all_data;
   /* mfem parameters */
   all_data.mfem.ref_levels = 4;
   all_data.mfem.order = 1;
   strcpy(all_data.mfem.mesh_file, "./mfem/data/ball-nurbs.mesh");

   all_data.input.test_problem = LAPLACE_2D5PT;
   all_data.input.tol = 1e-9;
   all_data.input.async_flag = 0;
   all_data.input.async_type = FULL_ASYNC;
   all_data.input.check_resnorm_flag = 1;
   all_data.input.global_conv_flag = 0;
   all_data.input.thread_part_type = ONE_LEVEL;
   all_data.input.converge_test_type = ONE_LEVEL;
   all_data.input.thread_part_distr_type = EQUAL_THREADS;
   all_data.input.num_pre_smooth_sweeps = 1;
   all_data.input.num_post_smooth_sweeps = 1;
   all_data.input.num_fine_smooth_sweeps = 1;
   all_data.input.num_coarse_smooth_sweeps = 1;
   all_data.input.format_output_flag = 0;
   all_data.input.num_threads = 1;
   all_data.input.print_reshist_flag = 1;
   all_data.input.smooth_weight = .8;
   all_data.input.smoother = JACOBI;
   all_data.input.solver = MULT;

   int num_cycles = 20;
   int start_cycle = num_cycles;
   int c = 1;

   /* Parse command line */
   int arg_index = 0;
   int print_usage = 0;

   while (arg_index < argc)
   {
      if (strcmp(argv[arg_index], "-n") == 0)
      {
         arg_index++;
         n = atoi(argv[arg_index]);
      }
      if (strcmp(argv[arg_index], "-problem") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "laplace_2d5pt") == 0){
            all_data.input.test_problem = LAPLACE_2D5PT;
         }
         else if (strcmp(argv[arg_index], "mfem") == 0){
            all_data.input.test_problem = MFEM;
         }
      }
      else if (strcmp(argv[arg_index], "-smoother") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "j") == 0){
            all_data.input.smoother = JACOBI;
         }
         else if (strcmp(argv[arg_index], "gs") == 0){
            all_data.input.smoother = GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "hybrid_jgs") == 0){
            all_data.input.smoother = HYBRID_JACOBI_GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "symm_j") == 0){
            all_data.input.smoother = SYMM_JACOBI;
         }
         else if (strcmp(argv[arg_index], "semi_async_gs") == 0){
            all_data.input.smoother = SEMI_ASYNC_GAUSS_SEIDEL;
         }
      }
      else if (strcmp(argv[arg_index], "-solver") == 0)
      {
         arg_index++;
         all_data.input.solver = atoi(argv[arg_index]);
         if (strcmp(argv[arg_index], "mult") == 0){
            all_data.input.solver = MULT;
         }
         else if (strcmp(argv[arg_index], "multadd") == 0){
            all_data.input.solver = MULTADD;
         }
         else if (strcmp(argv[arg_index], "afacx") == 0){
            all_data.input.solver = AFACX;
         }
         else if (strcmp(argv[arg_index], "async_multadd") == 0){
            all_data.input.solver = ASYNC_MULTADD;
            all_data.input.async_flag = 1;
            all_data.input.thread_part_type = ALL_LEVELS;
         }
         else if (strcmp(argv[arg_index], "async_afacx") == 0){
            all_data.input.solver = ASYNC_AFACX;
            all_data.input.async_flag = 1;
            all_data.input.thread_part_type = ALL_LEVELS;
         }
      }
      else if (strcmp(argv[arg_index], "-num_cycles") == 0)
      {
         arg_index++;
         num_cycles = start_cycle = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-start_cycle") == 0)
      {
         arg_index++;
         start_cycle = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-incr_cycle") == 0)
      {
         arg_index++;
         c = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-tol") == 0)
      {
         arg_index++;
         all_data.input.tol = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_smooth_sweeps") == 0)
      {
         arg_index++;
         all_data.input.num_pre_smooth_sweeps = atoi(argv[arg_index]);
         all_data.input.num_post_smooth_sweeps = atoi(argv[arg_index]);
         all_data.input.num_fine_smooth_sweeps = atoi(argv[arg_index]);
         all_data.input.num_coarse_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mxl") == 0)
      {
         arg_index++;
         max_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-agg_nl") == 0)
      {
         arg_index++;
         agg_num_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-coarsen_type") == 0)
      {
         arg_index++;
         coarsen_type = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_ref_levels") == 0)
      {
         arg_index++;
         all_data.mfem.ref_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_order") == 0)
      {
         arg_index++;
         all_data.mfem.order = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_mesh_file") == 0)
      {
         arg_index++;
         strcpy(all_data.mfem.mesh_file, argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-thread_level_part") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "one") == 0){
            all_data.input.thread_part_type = ONE_LEVEL;
         }
         else if (strcmp(argv[arg_index], "all") == 0){
            all_data.input.thread_part_type = ALL_LEVELS;
         }
      }
      else if (strcmp(argv[arg_index], "-converge_test_type") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "one") == 0){
            all_data.input.converge_test_type = ONE_LEVEL;
            all_data.input.global_conv_flag = 0;
         }
         else if (strcmp(argv[arg_index], "all") == 0){
            all_data.input.converge_test_type = ALL_LEVELS;
            all_data.input.global_conv_flag = 1;
         }
      }
      else if (strcmp(argv[arg_index], "-check_resnorm") == 0)
      {
         all_data.input.check_resnorm_flag = 1;
         all_data.input.global_conv_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-async_type") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "full") == 0){
            all_data.input.async_type = FULL_ASYNC;
         }
         else if (strcmp(argv[arg_index], "semi") == 0){
            all_data.input.async_type = SEMI_ASYNC;
         }
      }
      else if (strcmp(argv[arg_index], "-num_threads") == 0)
      {
         arg_index++;
         all_data.input.num_threads = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_runs") == 0)
      {
         arg_index++;
         num_runs = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-format_output") == 0)
      {
         all_data.input.format_output_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-hypre_print") == 0)
      {
         hypre_print_level = 3;
      }
      else if (strcmp(argv[arg_index], "-hypre_solve") == 0)
      {
         hypre_solve_flag = 3;
      }
      else if (strcmp(argv[arg_index], "-help") == 0)
      {
         print_usage = 1;
         break;
      }
      arg_index++;
   }

   if (all_data.input.solver == MULT){
      all_data.input.thread_part_type = ONE_LEVEL;
   }

   omp_set_num_threads(1);
   mkl_set_num_threads(1);

   if ((print_usage) && (myid == 0))
   {
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf("  -n <n>              : problem size in each direction (default: 33)\n");
      printf("  -solver <ID>        : solver ID\n");
      printf("                        0  - AMG (default) \n");
      printf("\n");
   }

   if (print_usage)
   {
      MPI_Finalize();
      return 0;
   }
   
   if (all_data.input.test_problem == LAPLACE_2D5PT){
      Laplacian_2D_5pt(&A, n);
   }
   else if (all_data.input.test_problem = MFEM){
      MFEM_Ex1(&all_data, &A);
   }

   HYPRE_IJMatrixAssemble(A);

   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);


  // HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b);
  // HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
  // HYPRE_IJVectorInitialize(b);

  // HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
  // HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
  // HYPRE_IJVectorInitialize(x);

  // double *rhs_values, *x_values;
  // int *rows;

  // rhs_values =  (double*) calloc(local_size, sizeof(double));
  // x_values =  (double*) calloc(local_size, sizeof(double));
  // rows = (int*) calloc(local_size, sizeof(int));

  // for (int i = 0; i < local_size; i++)
  // {
  //    rhs_values[i] = 1;
  //    x_values[i] = 0.0;
  //    rows[i] = ilower + i;
  // }

  // HYPRE_IJVectorSetValues(b, local_size, rows, rhs_values);
  // HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

  // free(x_values);
  // free(rhs_values);
  // free(rows);


  // HYPRE_IJVectorAssemble(b);
  // HYPRE_IJVectorGetObject(b, (void **) &par_b);

  // HYPRE_IJVectorAssemble(x);
  // HYPRE_IJVectorGetObject(x, (void **) &par_x);
   
   HYPRE_BoomerAMGCreate(&solver);

   HYPRE_BoomerAMGSetPrintLevel(solver, hypre_print_level);
   HYPRE_BoomerAMGSetOldDefault(solver);

   HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type);
   HYPRE_BoomerAMGSetMaxLevels(solver, max_levels);
   HYPRE_BoomerAMGSetAggNumLevels(solver, agg_num_levels);

   start = omp_get_wtime();
   HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
   all_data.output.hypre_setup_wtime = omp_get_wtime() - start;

  // if (hypre_solve_flag){
  //    HYPRE_BoomerAMGSetNumSweeps(solver, 1);
  //    HYPRE_BoomerAMGSetRelaxType(solver, 1);
  //    HYPRE_BoomerAMGSetTol(solver, 1e-7);
  //    HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
  //    int num_iterations;
  //    double final_res_norm;
  //    HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
  //    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
  //    if (all_data.input.format_output_flag){
  //       printf("HYPRE solve stats:\n");
  //       printf("\tIterations = %d\n", num_iterations);
  //       printf("\tRelative residual 2-norm = %e\n", final_res_norm);
  //    }
  //    else{
  //       printf("%d, %e\n", num_iterations, final_res_norm);
  //    }
  // }

   omp_set_num_threads(all_data.input.num_threads);
   start = omp_get_wtime();
   SMEM_Setup(solver, &all_data);
   all_data.output.setup_wtime = omp_get_wtime() - start; 

   for (int cycle = start_cycle; cycle <= num_cycles; cycle += c){   
      all_data.input.num_cycles = cycle;
      for (int run = 1; run <= num_runs; run++){
         InitSolve(&all_data);
         SMEM_Solve(&all_data);
         PrintOutput(all_data);
      }
   }

   HYPRE_BoomerAMGDestroy(solver);
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);
   
   MPI_Finalize();

   return 0;
}
