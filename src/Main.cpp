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
#include "SEQ_Setup.hpp"
#include "SMEM_Solve.hpp"


int main (int argc, char *argv[])
{
   int myid, num_procs;
   int N, n;

   int ilower, iupper;
   int local_size, extra;


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

   AllData all_data;
   all_data.input.num_pre_smooth_sweeps = 1;
   all_data.input.num_post_smooth_sweeps = 1;
   all_data.input.num_fine_smooth_sweeps = 1;
   all_data.input.num_coarse_smooth_sweeps = 1;
   all_data.input.num_cycles = 20;
   all_data.input.format_output_flag = 0;
   all_data.input.num_threads = 1;
   all_data.input.print_reshist_flag = 1;
   all_data.input.smooth_weight = 1;
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
         all_data.input.smoother = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-solver") == 0)
      {
         arg_index++;
         all_data.input.solver = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-help") == 0)
      {
         print_usage = 1;
         break;
      }
      arg_index++;
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

   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = (int)sqrt((double)num_procs) + 1;
   N = n*n; /* global number of rows */

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(A);

   Laplacian_2D_5pt(&A, n, N, ilower, iupper);

   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A);

   /* Note: for the testing of small problems, one may wish to read
      in a matrix in IJ format (for the format, see the output files
      from the -print_system option).
      In this case, one would use the following routine:
      HYPRE_IJMatrixRead( <filename>, MPI_COMM_WORLD,
                          HYPRE_PARCSR, &A );
      <filename>  = IJ.A.out to read in what has been printed out
      by -print_system (processor numbers are omitted).
      A call to HYPRE_IJMatrixRead is an *alternative* to the
      following sequence of HYPRE_IJMatrix calls:
      Create, SetObjectType, Initialize, SetValues, and Assemble
   */


   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);


   /* Create the rhs and solution */
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);

   double *rhs_values, *x_values;
   int    *rows;

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
   
   /* Create solver */
   HYPRE_BoomerAMGCreate(&solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetPrintLevel(solver, hypre_print_level);  /* print solve info + parameters */
   HYPRE_BoomerAMGSetOldDefault(solver); /* Falgout coarsening with modified classical interpolaiton */

   HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type);
   HYPRE_BoomerAMGSetMaxLevels(solver, max_levels);
   HYPRE_BoomerAMGSetAggNumLevels(solver, agg_num_levels);

   HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);

   SEQ_Setup(solver, &all_data);
   omp_set_num_threads(all_data.input.num_threads);
   SMEM_Solve(&all_data);

  // HYPRE_BoomerAMGSetRelaxType(solver, 1);   /* G-S/Jacobi hybrid relaxation */
  // HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */
  // HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);

   HYPRE_BoomerAMGDestroy(solver);

   /* Clean up */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
