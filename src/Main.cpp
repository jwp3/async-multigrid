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


int main (int argc, char *argv[])
{
   int myid, num_procs;
   int N, n;

   int ilower, iupper;
   int local_size, extra;

   double start;

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
   all_data.input.async_flag = 0;
   all_data.input.thread_part_type = ONE_LEVEL;
   all_data.input.num_pre_smooth_sweeps = 1;
   all_data.input.num_post_smooth_sweeps = 1;
   all_data.input.num_fine_smooth_sweeps = 1;
   all_data.input.num_coarse_smooth_sweeps = 1;
   all_data.input.num_cycles = 20;
   all_data.input.format_output_flag = 0;
   all_data.input.num_threads = 1;
   all_data.input.print_reshist_flag = 1;
   all_data.input.smooth_weight = .8;
   all_data.input.smoother = JACOBI;
   all_data.input.solver = MULT;


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
      }
      else if (strcmp(argv[arg_index], "-solver") == 0)
      {
         arg_index++;
         all_data.input.solver = atoi(argv[arg_index]);
         if (strcmp(argv[arg_index], "mult") == 0){
            all_data.input.solver = MULT;
         }
         else if (strcmp(argv[arg_index], "mult_add") == 0){
            all_data.input.solver = MULT_ADD;
         }
         else if (strcmp(argv[arg_index], "afacx") == 0){
            all_data.input.solver = AFACX;
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
         all_data.input.num_cycles = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-num_threads") == 0)
      {
         arg_index++;
         all_data.input.num_threads = atoi(argv[arg_index]);
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
      return (0);
   }

   if (n*n < num_procs) n = (int)sqrt((double)num_procs) + 1;
   N = n*n;

   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   local_size = iupper - ilower + 1;

   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

   HYPRE_IJMatrixInitialize(A);

   Laplacian_2D_5pt(&A, n, N, ilower, iupper);

   HYPRE_IJMatrixAssemble(A);

   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);


   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);

   double *rhs_values, *x_values;
   int *rows;

   rhs_values =  (double*) calloc(local_size, sizeof(double));
   x_values =  (double*) calloc(local_size, sizeof(double));
   rows = (int*) calloc(local_size, sizeof(int));

   for (int i = 0; i < local_size; i++)
   {
      rhs_values[i] = 1;
      x_values[i] = 0.0;
      rows[i] = ilower + i;
   }

   HYPRE_IJVectorSetValues(b, local_size, rows, rhs_values);
   HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

   free(x_values);
   free(rhs_values);
   free(rows);


   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject(b, (void **) &par_b);

   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);
   
   HYPRE_BoomerAMGCreate(&solver);

   HYPRE_BoomerAMGSetPrintLevel(solver, hypre_print_level);
   HYPRE_BoomerAMGSetOldDefault(solver);

   HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type);
   HYPRE_BoomerAMGSetMaxLevels(solver, max_levels);
   HYPRE_BoomerAMGSetAggNumLevels(solver, agg_num_levels);

   start = omp_get_wtime();
   HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
   all_data.output.hypre_setup_wtime = omp_get_wtime() - start;

   if (hypre_solve_flag){
      HYPRE_BoomerAMGSetNumSweeps(solver, 1);
      HYPRE_BoomerAMGSetRelaxType(solver, 1);
      HYPRE_BoomerAMGSetTol(solver, 1e-7);
      HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
      int num_iterations;
      double final_res_norm;
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (all_data.input.format_output_flag){
         printf("HYPRE solve stats:\n");
         printf("\tIterations = %d\n", num_iterations);
         printf("\tRelative residual 2-norm = %e\n", final_res_norm);
      }
      else{
         printf("%d, %e\n", num_iterations, final_res_norm);
      }
   }

   omp_set_num_threads(all_data.input.num_threads);
  // omp_set_num_threads(1);
   start = omp_get_wtime();
   SMEM_Setup(solver, &all_data);
   all_data.output.setup_wtime = omp_get_wtime() - start;
   start = omp_get_wtime();
   SMEM_Solve(&all_data);
   all_data.output.solve_wtime = omp_get_wtime() - start;

   char print_str[1000];
   if (all_data.input.format_output_flag == 0){
      strcpy(print_str, "\nSetup stats:\n"
                        "\tHypre setup time = %e\n"
                        "\tRemaining setup time = %e\n"
                        "\tTotal setup time = %e\n"
                        "\nSolve stats:\n"
                        "\tTotal solve time = %e\n");
   }
   else{
      strcpy(print_str, "%e %e %e %e\n");
   }

   printf(print_str,
          all_data.output.hypre_setup_wtime,
          all_data.output.setup_wtime,
          all_data.output.hypre_setup_wtime + all_data.output.hypre_setup_wtime,
          all_data.output.solve_wtime);

   HYPRE_BoomerAMGDestroy(solver);

   /* Clean up */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
