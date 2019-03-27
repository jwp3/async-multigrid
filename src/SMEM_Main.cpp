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
#include "Elasticity.hpp"
#include "Maxwell.hpp"
#include "SEQ_MatVec.hpp"
#include "SEQ_AMG.hpp"
#include "SMEM_Setup.hpp"
#include "SMEM_Solve.hpp"
#include "Misc.hpp"


int main (int argc, char *argv[])
{
   int myid, num_procs;

   double start;
   int num_runs = 1;
   int time_barrier_flag = 0;

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

   int n = 10;
   int nx = n, ny = n, nz = n;
   /* Hypre parameters */
   int max_levels = 20;
   int solver_id = 0;
   HYPRE_Int agg_num_levels = 0;
   HYPRE_Int coarsen_type = 10;
   HYPRE_Int interp_type = 6;
   int hypre_print_level = 0;
   int hypre_solve_flag = 0;
   double strong_threshold = .25;

   AllData all_data;
   /* mfem parameters */
   all_data.mfem.ref_levels = 4;
   all_data.mfem.order = 1;
   strcpy(all_data.mfem.mesh_file, "./mfem/data/ball-nurbs.mesh");
   all_data.mfem.amr_refs = 0;

   all_data.input.test_problem = LAPLACE_2D5PT;
   all_data.input.tol = 1e-9;
   all_data.input.async_flag = 0;
   all_data.input.async_type = FULL_ASYNC;
   all_data.input.check_resnorm_flag = 0;
   all_data.input.global_conv_flag = 0;
   all_data.input.thread_part_type = ALL_LEVELS;
   all_data.input.converge_test_type = LOCAL;
   all_data.input.res_compute_type = LOCAL;
   all_data.input.thread_part_distr_type = BALANCED_THREADS;
   all_data.input.num_pre_smooth_sweeps = 1;
   all_data.input.num_post_smooth_sweeps = 1;
   all_data.input.num_fine_smooth_sweeps = 1;
   all_data.input.num_coarse_smooth_sweeps = 1;
   all_data.input.format_output_flag = 0;
   all_data.input.num_threads = 1;
   all_data.input.print_output_flag = 1;
   all_data.input.smooth_weight = 1;
   all_data.input.smoother = JACOBI;
   all_data.input.smooth_interp_type = JACOBI;
   all_data.input.solver = MULT;
   all_data.input.hypre_test_error_flag = 0;
   all_data.input.mfem_test_error_flag = 0;
   all_data.input.mfem_solve_print_flag = 0;
   all_data.input.sim_grid_wait = 0;
   all_data.input.sim_read_delay = 0;
   all_data.input.print_grid_wait_flag = 0;
   all_data.input.print_level_stats_flag = 0;
   all_data.input.print_reshist_flag = 0;
   all_data.input.read_type = READ_SOL;

   int num_cycles = 20;
   int start_cycle = num_cycles;
   int c = 1;
   int warmup = 1;

   /* Parse command line */
   int arg_index = 0;
   int print_usage = 0;

   while (arg_index < argc)
   {
      if (strcmp(argv[arg_index], "-n") == 0)
      {
         arg_index++;
	 n = atoi(argv[arg_index]);
         nx = ny = nz = n;
      }
      else if (strcmp(argv[arg_index], "-nx") == 0)
      {
         arg_index++;
         nx = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-ny") == 0)
      {
         arg_index++;
         ny = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-nz") == 0)
      {
         arg_index++;
         nz = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-problem") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "5pt") == 0){
            all_data.input.test_problem = LAPLACE_2D5PT;
         }
         else if (strcmp(argv[arg_index], "27pt") == 0){
            all_data.input.test_problem = LAPLACE_3D27PT;
         }
	 else if (strcmp(argv[arg_index], "7pt") == 0){
            all_data.input.test_problem = LAPLACE_3D7PT;
         }
         else if (strcmp(argv[arg_index], "mfem_laplace") == 0){
            all_data.input.test_problem = MFEM_LAPLACE;
         }
         else if (strcmp(argv[arg_index], "mfem_elast") == 0){
            all_data.input.test_problem = MFEM_ELAST;
         }
	 else if (strcmp(argv[arg_index], "mfem_maxwell") == 0){
            all_data.input.test_problem = MFEM_MAXWELL;
         }
      }
      else if (strcmp(argv[arg_index], "-smoother") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "j") == 0){
            all_data.input.smoother = JACOBI;
            all_data.input.smooth_interp_type = JACOBI;
         }
         else if (strcmp(argv[arg_index], "gs") == 0){
            all_data.input.smoother = GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "hybrid_jgs") == 0){
            all_data.input.smoother = HYBRID_JACOBI_GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "async_gs") == 0){
            all_data.input.smoother = ASYNC_GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "semi_async_gs") == 0){
            all_data.input.smoother = SEMI_ASYNC_GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "L1j") == 0){
            all_data.input.smoother = L1_JACOBI;
            all_data.input.smooth_interp_type = L1_JACOBI;
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
         }
         else if (strcmp(argv[arg_index], "async_afacx") == 0){
            all_data.input.solver = ASYNC_AFACX;
            all_data.input.async_flag = 1;
         }
      }
      else if (strcmp(argv[arg_index], "-smooth_interp") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "j") == 0){
            all_data.input.smooth_interp_type = JACOBI;
         }
         else if (strcmp(argv[arg_index], "L1j") == 0){
            all_data.input.smooth_interp_type = L1_JACOBI;
         }
      }
      else if (strcmp(argv[arg_index], "-smooth_weight") == 0)
      {
         arg_index++;
         all_data.input.smooth_weight = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-th") == 0)
      {
         arg_index++;
         strong_threshold = atof(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-num_pre_smooth_sweeps") == 0)
      {
         arg_index++;
         all_data.input.num_pre_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_post_smooth_sweeps") == 0)
      {
         arg_index++;
         all_data.input.num_post_smooth_sweeps = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-interp_type") == 0)
      {
         arg_index++;
         interp_type = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_ref_levels") == 0)
      {
         arg_index++;
         all_data.mfem.ref_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_amr_refs") == 0)
      {
         arg_index++;
         all_data.mfem.amr_refs = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-thread_part_type") == 0)
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
         if (strcmp(argv[arg_index], "local") == 0){
            all_data.input.converge_test_type = LOCAL;
            all_data.input.global_conv_flag = 0;
         }
         else if (strcmp(argv[arg_index], "global") == 0){
            all_data.input.converge_test_type = GLOBAL;
            all_data.input.global_conv_flag = 1;
         }
      }
      else if (strcmp(argv[arg_index], "-res_compute_type") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "local") == 0){
            all_data.input.res_compute_type = LOCAL;
         }
         else if (strcmp(argv[arg_index], "global") == 0){
            all_data.input.res_compute_type = GLOBAL;
         }
      }
     // else if (strcmp(argv[arg_index], "-check_resnorm") == 0)
     // {
     //    all_data.input.check_resnorm_flag = 1;
     // }
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
      else if (strcmp(argv[arg_index], "-sim_grid_wait") == 0)
      {
         arg_index++;
         all_data.input.sim_grid_wait = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-sim_read_delay") == 0)
      {
         arg_index++;
         all_data.input.sim_read_delay = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-read_type") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "sol") == 0){
            all_data.input.read_type = READ_SOL;
         }
         else if (strcmp(argv[arg_index], "res") == 0){
            all_data.input.read_type = READ_RES;
         }
      }
      else if (strcmp(argv[arg_index], "-no_output") == 0)
      {
         all_data.input.print_output_flag = 0;
      }
      else if (strcmp(argv[arg_index], "-format_output") == 0)
      {
         all_data.input.format_output_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_reshist") == 0)
      {
         all_data.input.print_reshist_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_hypre") == 0)
      {
         hypre_print_level = 3;
      }
      else if (strcmp(argv[arg_index], "-print_grid_wait") == 0)
      {
         all_data.input.print_grid_wait_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_level_stats") == 0)
      {
         all_data.input.print_level_stats_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-hypre_test_error") == 0)
      {
         all_data.input.hypre_test_error_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-mfem_test_error") == 0)
      {
         all_data.input.mfem_test_error_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-time_barrier") == 0)
      {
         time_barrier_flag = 1;
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
   else{
      all_data.input.thread_part_type = ALL_LEVELS;
   }
   if (all_data.input.solver == MULT ||
       all_data.input.solver == MULTADD ||
       all_data.input.solver == AFACX ||
       all_data.input.solver == ASYNC_AFACX){
      all_data.input.res_compute_type = LOCAL; 
   }
   
   srand(0);
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
 
   start = omp_get_wtime(); 
   if (all_data.input.test_problem == LAPLACE_2D5PT){
      Laplacian_2D_5pt(&A, n);
      HYPRE_IJMatrixAssemble(A);
      HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   }
   else if (all_data.input.test_problem == LAPLACE_3D27PT){
      Laplacian_3D_27pt(&parcsr_A, nx, ny, nz);
   }
   else if (all_data.input.test_problem == LAPLACE_3D7PT){
      Laplacian_3D_7pt(&parcsr_A, nx, ny, nz);
   }
   else if (all_data.input.test_problem == MFEM_LAPLACE){
      MFEM_Laplacian(&all_data, &A);
      HYPRE_IJMatrixAssemble(A);
      HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   }
   else if (all_data.input.test_problem == MFEM_ELAST){
      MFEM_Elasticity(&all_data, &A);
      HYPRE_IJMatrixAssemble(A);
      HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   }
   else if (all_data.input.test_problem == MFEM_MAXWELL){
      MFEM_Maxwell(&all_data, &A);
      HYPRE_IJMatrixAssemble(A);
      HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   }
   else{
      return 1;
   }
   all_data.output.prob_setup_wtime = omp_get_wtime() - start;

   start = omp_get_wtime();
   HYPRE_BoomerAMGCreate(&solver);

   HYPRE_BoomerAMGSetPrintLevel(solver, hypre_print_level);
   HYPRE_BoomerAMGSetOldDefault(solver);

   HYPRE_BoomerAMGSetPostInterpType(solver, 0);
   HYPRE_BoomerAMGSetInterpType(solver, interp_type);
   HYPRE_BoomerAMGSetRestriction(solver, 0);
   HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type);
   HYPRE_BoomerAMGSetMaxLevels(solver, max_levels);
   HYPRE_BoomerAMGSetAggNumLevels(solver, agg_num_levels);

   HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
   all_data.output.hypre_setup_wtime = omp_get_wtime() - start;

   start = omp_get_wtime();
   omp_set_num_threads(all_data.input.num_threads);
   SMEM_Setup(solver, &all_data);
   all_data.output.setup_wtime = omp_get_wtime() - start; 

   if (all_data.input.hypre_test_error_flag == 1){
      int relax_type;
      if (all_data.input.smoother == L1_JACOBI){
         relax_type = 16;
      }
      else if (all_data.input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL ||
	       all_data.input.smoother == GAUSS_SEIDEL){
         relax_type = 3;
      }
      else{
         relax_type = 0;
      }
      HYPRE_ParCSRHybridSetStrongThreshold(solver, strong_threshold);
      HYPRE_BoomerAMGSetRelaxType(solver, 0);
      HYPRE_BoomerAMGSetRelaxWt(solver, all_data.input.smooth_weight);
      HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, all_data.grid.n[0]-1, &b);
      HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(b);

      HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, all_data.grid.n[0]-1, &x);
      HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(x);

      double *rhs_values, *x_values;
      int    *rows;

      rhs_values =  (double*) calloc(all_data.grid.n[0], sizeof(double));
      x_values =  (double*) calloc(all_data.grid.n[0], sizeof(double));
      rows = (int*) calloc(all_data.grid.n[0], sizeof(int));

      for (int i = 0; i < all_data.grid.n[0]; i++){
         rhs_values[i] = 1.0;
         x_values[i] = 0.0;
         rows[i] = i;
      }

      HYPRE_IJVectorSetValues(b, all_data.grid.n[0], rows, rhs_values);
      HYPRE_IJVectorSetValues(x, all_data.grid.n[0], rows, x_values);

      free(x_values);
      free(rhs_values);
      free(rows);

      HYPRE_IJVectorAssemble(b);
      HYPRE_IJVectorGetObject(b, (void **)&par_b);
      HYPRE_IJVectorAssemble(x);
      HYPRE_IJVectorGetObject(x, (void **)&par_x);

      HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
   }

  // if (time_barrier_flag == 1 && all_data.input.thread_part_type == ONE_LEVEL){
  //    double start, barrier_wtime;
  //    all_data.barrier.counter = 0;
  //    all_data.barrier.flag = 0;
  //    omp_init_lock(&(all_data.barrier.lock));
  //    start = omp_get_wtime();
  //    #pragma omp parallel
  //    {
  //       for (int cycle = start_cycle; cycle <= num_cycles; cycle += c){
  //          SMEM_LevelBarrier(&all_data, all_data.thread.barrier_flags, 0);
  //         // SMEM_SRCLevelBarrier(&all_data, &(all_data.barrier.flag), 0);
  //         // #pragma omp flush(all_data)
  //       }
  //    }
  //    printf("my time = %e\n", omp_get_wtime() - start);
  //    omp_destroy_lock(&(all_data.barrier.lock));
  //    start = omp_get_wtime();
  //    #pragma omp parallel
  //    {
  //       for (int cycle = start_cycle; cycle <= num_cycles; cycle += c){
  //          #pragma omp barrier
  //          SMEM_LevelBarrier(&all_data, all_data.thread.barrier_flags, 0);
  //       }
  //    }
  //    printf("OpenMP time = %e\n", omp_get_wtime() - start);
  //    return 0;
  // }

   srand(time(NULL));
   if (warmup == 1){
      num_runs++;
   }
   for (int cycle = start_cycle; cycle <= num_cycles; cycle += c){   
      all_data.input.num_cycles = cycle;
      for (int run = 1; run <= num_runs; run++){
         InitSolve(&all_data);
         SMEM_Solve(&all_data);
         if (all_data.input.mfem_test_error_flag == 1){
            for (int i = 0; i < all_data.grid.n[0]; i++){
               all_data.output.mfem_e_norm2 += pow(all_data.mfem.u[i] - all_data.vector.u[0][i], 2.0);
	      // printf("%e, %e, %e, %e\n",
              //        all_data.mfem.u[i], par_x->local_vector->data[i], all_data.vector.u[0][i],
              //        all_data.mfem.u[i] - par_x->local_vector->data[i]);
            }
         }
         if (all_data.input.hypre_test_error_flag == 1){
            for (int i = 0; i < all_data.grid.n[0]; i++){
               all_data.output.hypre_e_norm2 += pow(par_x->local_vector->data[i] - all_data.vector.u[0][i], 2.0);
            }
         }
	 if (warmup == 1){
	    if (all_data.input.print_output_flag == 1 && run > 1){
               PrintOutput(all_data);
            }
	 }
	 else {
	    if (all_data.input.print_output_flag == 1){
	       PrintOutput(all_data);
	    }
	 }
      }
   }

  // HYPRE_BoomerAMGDestroy(solver);
  // HYPRE_IJMatrixDestroy(A);
  // HYPRE_IJVectorDestroy(b);
  // HYPRE_IJVectorDestroy(x);
   
   MPI_Finalize();

   return 0;
}
