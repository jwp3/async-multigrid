#include "Main.hpp"
#include "Laplacian.hpp"
#include "Elasticity.hpp"
#include "Maxwell.hpp"
#include "SEQ_MatVec.hpp"
#include "SEQ_AMG.hpp"
#include "SMEM_Setup.hpp"
#include "SMEM_Solve.hpp"
#include "SMEM_ExtendedSystem.hpp"
#include "Misc.hpp"


int main (int argc, char *argv[])
{
   AllData all_data;
   int myid, num_procs;

   double start;
   int num_runs = 1;
   int time_barrier_flag = 0;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   all_data.matrix.n = 10;
   all_data.matrix.nx = all_data.matrix.n;
   all_data.matrix.ny = all_data.matrix.n;
   all_data.matrix.nz = all_data.matrix.n;
   /* Hypre parameters */
   all_data.hypre.max_levels = 25;
   all_data.hypre.agg_num_levels = 0;
   all_data.hypre.coarsen_type = 9;
   all_data.hypre.interp_type = 6;
   all_data.hypre.print_level = 0;
   all_data.hypre.strong_threshold = .25;
   all_data.hypre.num_functions = 1;
   all_data.hypre.P_max_elmts = 0;
   int solver_id = 0;
   int hypre_solve_flag = 0;
   int hypre_num_threads = 1;

   /* mfem parameters */
   all_data.mfem.ref_levels = 1;
   all_data.mfem.order = 1;
   all_data.mfem.amr_refs = 0;
   all_data.mfem.max_amr_dofs = (int)pow((double)all_data.matrix.n, 3.0);

   all_data.input.test_problem = LAPLACE_2D5PT;
   all_data.input.tol = 1e-9;
   all_data.input.async_flag = 0;
   all_data.input.async_type = FULL_ASYNC;
   all_data.input.check_resnorm_flag = 1;
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
   all_data.input.eig_power_max_iters = 50;
   all_data.input.cheby_flag = 0;
   all_data.input.precond_flag = 0;
   all_data.input.delay_usec = 0;
   all_data.input.delay_flag = DELAY_NONE;
   all_data.input.omp_parfor_flag = 1;
   all_data.input.delay_frac = 0.0;
   all_data.input.construct_R_flag = 0;

   int num_cycles = 20;
   int start_cycle = num_cycles;
   int c = 1;
   int warmup = 0;
   int only_setup_flag = 0;
   int strongscale_flag = 0;

   /* Parse command line */
   int arg_index = 0;
   int print_usage = 0;

   while (arg_index < argc)
   {
      if (strcmp(argv[arg_index], "-n") == 0)
      {
         arg_index++;
	 all_data.matrix.n = atoi(argv[arg_index]);
         all_data.matrix.nx = all_data.matrix.ny = all_data.matrix.nz = all_data.matrix.n;
        // all_data.mfem.max_amr_dofs = (int)pow((double)all_data.matrix.n, 3.0);
         all_data.mfem.max_amr_dofs = all_data.matrix.nx * all_data.matrix.ny * all_data.matrix.nz;
      }
      else if (strcmp(argv[arg_index], "-nx") == 0)
      {
         arg_index++;
         all_data.matrix.nx = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-ny") == 0)
      {
         arg_index++;
         all_data.matrix.ny = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-nz") == 0)
      {
         arg_index++;
         all_data.matrix.nz = atoi(argv[arg_index]);
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
            strcpy(all_data.mfem.mesh_file, "./mfem_haswell/mfem-4.0/data/ball-nurbs.mesh");
         }
         else if (strcmp(argv[arg_index], "mfem_elast") == 0){
            all_data.input.test_problem = MFEM_ELAST;
            strcpy(all_data.mfem.mesh_file, "./mfem_haswell/mfem-4.0/data/beam-hex.mesh");
            all_data.hypre.num_functions = 3;
         }
	 else if (strcmp(argv[arg_index], "mfem_maxwell") == 0){
            all_data.input.test_problem = MFEM_MAXWELL;
         }
         else if (strcmp(argv[arg_index], "file") == 0){
            all_data.input.test_problem = MATRIX_FROM_FILE;
            arg_index++;
            strcpy(all_data.input.mat_file_str, argv[arg_index]);
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
         else if (strcmp(argv[arg_index], "bpx") == 0){
            all_data.input.solver = BPX;
         }
         else if (strcmp(argv[arg_index], "par_bpx") == 0){
            all_data.input.solver = PAR_BPX;
         }
         else if (strcmp(argv[arg_index], "eebpx") == 0){
            all_data.input.solver = EXPLICIT_EXTENDED_SYSTEM_BPX;
         }
         else if (strcmp(argv[arg_index], "async_eebpx") == 0){
            all_data.input.solver = EXPLICIT_EXTENDED_SYSTEM_BPX; 
            all_data.input.async_flag = 1;
         }
         else if (strcmp(argv[arg_index], "iebpx") == 0){
            all_data.input.solver = IMPLICIT_EXTENDED_SYSTEM_BPX;
         }
         else if (strcmp(argv[arg_index], "async_iebpx") == 0){
            all_data.input.solver = IMPLICIT_EXTENDED_SYSTEM_BPX;
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
         all_data.hypre.strong_threshold = atof(argv[arg_index]);
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
         all_data.hypre.max_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-agg_nl") == 0)
      {
         arg_index++;
         all_data.hypre.agg_num_levels = atoi(argv[arg_index]);
      }
      else if ( strcmp(argv[arg_index], "-Pmx") == 0 ){
         arg_index++;
         all_data.hypre.P_max_elmts = atoi(argv[arg_index]);
      } 
      else if (strcmp(argv[arg_index], "-coarsen_type") == 0)
      {
         arg_index++;
         all_data.hypre.coarsen_type = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-interp_type") == 0)
      {
         arg_index++;
         all_data.hypre.interp_type = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-no_resnorm") == 0)
      {
         all_data.input.check_resnorm_flag = 0;
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
      else if (strcmp(argv[arg_index], "-hypre_num_threads") == 0)
      {
         arg_index++;
         hypre_num_threads = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-oneline_output") == 0)
      {
         all_data.input.format_output_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_reshist") == 0)
      {
         all_data.input.print_reshist_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_hypre") == 0)
      {
         all_data.hypre.print_level = 3;
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
      else if (strcmp(argv[arg_index], "-cheby") == 0)
      {
         all_data.input.cheby_flag = 1;
         all_data.input.precond_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-only_setup") == 0){
         only_setup_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-eig_power_max_iters") == 0){
         arg_index++;
         all_data.input.eig_power_max_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-warmup") == 0){
         warmup = 1;
      }
      else if (strcmp(argv[arg_index], "-delay_one") == 0){
         arg_index++;
         all_data.input.delay_usec = atoi(argv[arg_index]);
         all_data.input.delay_flag = DELAY_ONE;
      }
      else if (strcmp(argv[arg_index], "-delay_some") == 0){
         arg_index++;
         all_data.input.delay_usec = atoi(argv[arg_index]);
         arg_index++;
         all_data.input.delay_frac = atof(argv[arg_index]);
         all_data.input.delay_flag = DELAY_SOME;
      }
      else if (strcmp(argv[arg_index], "-delay_all") == 0){
         arg_index++;
         all_data.input.delay_usec = atoi(argv[arg_index]);
         all_data.input.delay_flag = DELAY_ALL;
         all_data.input.delay_frac = 1.0;
      }
      else if (strcmp(argv[arg_index], "-delay_all") == 0){
         arg_index++;
         all_data.input.delay_usec = atoi(argv[arg_index]);
         all_data.input.delay_flag = DELAY_ALL;
         all_data.input.delay_frac = 1.0;
      }
      else if (strcmp(argv[arg_index], "-construct_R") == 0){
         all_data.input.construct_R_flag = 1;
      }
      arg_index++;
   }

   if (all_data.input.solver == MULT ||
       all_data.input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX ||
       all_data.input.solver == BPX ||
       all_data.input.solver == PAR_BPX){
      all_data.input.thread_part_type = ONE_LEVEL;
   }
   else{
      all_data.input.thread_part_type = ALL_LEVELS;
   }

   if (all_data.input.solver == MULT ||
       all_data.input.solver == MULTADD ||
       all_data.input.solver == AFACX ||
       all_data.input.solver == ASYNC_AFACX ||
       all_data.input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX ||
       all_data.input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX ||
       all_data.input.solver == BPX ||
       all_data.input.solver == PAR_BPX){
      all_data.input.res_compute_type = LOCAL; 
   }
   
   srand(0);
   omp_set_num_threads(1);
  // mkl_set_num_threads(1);

   start = omp_get_wtime();
   omp_set_num_threads(hypre_num_threads);
  // omp_set_num_threads(all_data.input.num_threads);
   SMEM_Setup(&all_data);
   all_data.output.setup_wtime = omp_get_wtime() - start; 
   if (only_setup_flag == 1){
      MPI_Finalize();
      return 0;
   }
   omp_set_num_threads(all_data.input.num_threads);

//   vector<int> t_vec;
//   if (strongscale_flag == 1){
//      t_vec.push_back(1);
//      t_vec.push_back(2);
//      int t = 4;
//      while (1){
//         t_vec.push_back(t);
//         if (t == 36) break;
//         t += 4;
//      }
//   }
//   else {
//      t_vec.push_back(all_data.input.num_threads);
//   }

//   if (all_data.input.hypre_test_error_flag == 1){
//      int relax_type;
//      if (all_data.input.smoother == L1_JACOBI){
//         relax_type = 16;
//      }
//      else if (all_data.input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL ||
//	       all_data.input.smoother == GAUSS_SEIDEL){
//         relax_type = 3;
//      }
//      else{
//         relax_type = 0;
//      }
//      HYPRE_ParCSRHybridSetStrongThreshold(solver, strong_threshold);
//      HYPRE_BoomerAMGSetRelaxType(solver, 0);
//      HYPRE_BoomerAMGSetRelaxWt(solver, all_data.input.smooth_weight);
//      HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, all_data.grid.n[0]-1, &b);
//      HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
//      HYPRE_IJVectorInitialize(b);
//
//      HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, all_data.grid.n[0]-1, &x);
//      HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
//      HYPRE_IJVectorInitialize(x);
//
//      double *rhs_values, *x_values;
//      int    *rows;
//
//      rhs_values =  (double*) calloc(all_data.grid.n[0], sizeof(double));
//      x_values =  (double*) calloc(all_data.grid.n[0], sizeof(double));
//      rows = (int*) calloc(all_data.grid.n[0], sizeof(int));
//
//      for (int i = 0; i < all_data.grid.n[0]; i++){
//         rhs_values[i] = 1.0;
//         x_values[i] = 0.0;
//         rows[i] = i;
//      }
//
//      HYPRE_IJVectorSetValues(b, all_data.grid.n[0], rows, rhs_values);
//      HYPRE_IJVectorSetValues(x, all_data.grid.n[0], rows, x_values);
//
//      free(x_values);
//      free(rhs_values);
//      free(rows);
//
//      HYPRE_IJVectorAssemble(b);
//      HYPRE_IJVectorGetObject(b, (void **)&par_b);
//      HYPRE_IJVectorAssemble(x);
//      HYPRE_IJVectorGetObject(x, (void **)&par_x);
//
//      HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
//   }

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
         if (all_data.input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX ||
             all_data.input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
            SMEM_ExtendedSystemSolve(&all_data);
         }
        // else if (all_data.input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
        //    SMEM_ImplicitExtendedSystemSolve(&all_data);
        // }
         else {
            SMEM_Solve(&all_data);
           // if (all_data.input.mfem_test_error_flag == 1){
           //    for (int i = 0; i < all_data.grid.n[0]; i++){
           //       all_data.output.mfem_e_norm2 += pow(all_data.mfem.u[i] - all_data.vector.u[0][i], 2.0);
           //      // printf("%e, %e, %e, %e\n",
           //      //        all_data.mfem.u[i], par_x->local_vector->data[i], all_data.vector.u[0][i],
           //      //        all_data.mfem.u[i] - par_x->local_vector->data[i]);
           //    }
           // }
           // if (all_data.input.hypre_test_error_flag == 1){
           //    for (int i = 0; i < all_data.grid.n[0]; i++){
           //       all_data.output.hypre_e_norm2 += pow(par_x->local_vector->data[i] - all_data.vector.u[0][i], 2.0);
           //    }
           // }
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
