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

   /* Hypre parameters */
   all_data.hypre.max_levels = 25;
   all_data.hypre.agg_num_levels = 0;
   all_data.hypre.coarsen_type = 10;
   all_data.hypre.interp_type = 6;
   all_data.hypre.print_level = 0;
   all_data.hypre.strong_threshold = .25;
   all_data.hypre.num_functions = 1;
   all_data.hypre.P_max_elmts = 0;
   all_data.input.simple_jacobi_flag = -1;
   int solver_id = 0;
   int hypre_solve_flag = 0;
   int hypre_num_threads = 1;
   all_data.input.hypre_memory = HYPRE_MEMORY_HOST;

   int n = 40;
   double c = 1.0, a = 1.0;
   all_data.matrix.nx = n;
   all_data.matrix.ny = n;
   all_data.matrix.nz = n;
   all_data.matrix.difconv_ax = a;
   all_data.matrix.difconv_ay = a;
   all_data.matrix.difconv_az = a;
   all_data.matrix.difconv_cx = c;
   all_data.matrix.difconv_cy = c;
   all_data.matrix.difconv_cz = c;
   all_data.matrix.difconv_atype = -1;
   all_data.matrix.vardifconv_eps = 1.0;

   /* mfem parameters */
   all_data.mfem.ref_levels = 1;
   all_data.mfem.order = 1;
   all_data.mfem.amr_refs = 0;
   all_data.mfem.max_amr_dofs = (int)pow((double)all_data.matrix.n, 3.0);
   strcpy(all_data.mfem.mesh_file, "./mfem_haswell/mfem-3.4/data/beam-tri.mesh");

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
   all_data.hypre.relax_type = 0;
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
   all_data.input.cheby_eig_max_iters = 20;
   all_data.input.cheby_eig_tol = 1e-9;
   all_data.input.cheby_flag = 0;
   all_data.input.precond_flag = 0;
   all_data.input.delay_usec = 0;
   all_data.input.delay_type = DELAY_NONE;
   all_data.input.omp_parfor_flag = 1;
   all_data.input.delay_frac = 0.0;
   all_data.input.construct_R_flag = 1;
   all_data.input.hypre_solver_flag = 0;
   all_data.input.fail_iter = -1;
   all_data.input.cheby_eig_type = CHEBY_EIG_POWER;
   all_data.input.rhs_type = RHS_RAND;

   int max_num_iters = 20;
   int start_num_iters = max_num_iters;
   int incr_num_iters = 1;
   int warmup = 0;
   int only_setup_flag = 0;
   int strongscale_flag = 0;

   /* Parse command line */
   int arg_index = 0;
   int print_usage = 0;
   int background_program = 0;

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
      else if (strcmp(argv[arg_index], "-c") == 0){
         arg_index++;
         c = atof(argv[arg_index]);
         all_data.matrix.difconv_cx = c;
         all_data.matrix.difconv_cy = c;
         all_data.matrix.difconv_cz = c;
      }
      else if (strcmp(argv[arg_index], "-cx") == 0){
         arg_index++;
         all_data.matrix.difconv_cx = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-cy") == 0){
         arg_index++;
         all_data.matrix.difconv_cy = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-cz") == 0){
         arg_index++;
         all_data.matrix.difconv_cz = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-a") == 0){
         arg_index++;
         a = atof(argv[arg_index]);
         all_data.matrix.difconv_ax = a;
         all_data.matrix.difconv_ay = a;
         all_data.matrix.difconv_az = a;
      }
      else if (strcmp(argv[arg_index], "-ax") == 0){
         arg_index++;
         all_data.matrix.difconv_ax = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-ay") == 0){
         arg_index++;
         all_data.matrix.difconv_ay = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-az") == 0){
         arg_index++;
         all_data.matrix.difconv_az = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-atype") == 0){
         arg_index++;
         all_data.matrix.difconv_atype = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-problem") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "5pt") == 0){
            all_data.input.test_problem = LAPLACE_2D5PT;
         }
         else if (strcmp(argv[arg_index], "27pt") == 0){
            all_data.input.test_problem = LAPLACE_3D27PT;
            all_data.matrix.difconv_ax = 0.0;
            all_data.matrix.difconv_ay = 0.0;
            all_data.matrix.difconv_az = 0.0;
         }
	 else if (strcmp(argv[arg_index], "7pt") == 0){
            all_data.input.test_problem = LAPLACE_3D7PT;
            all_data.matrix.difconv_ax = 0.0;
            all_data.matrix.difconv_ay = 0.0;
            all_data.matrix.difconv_az = 0.0;
         }
         else if (strcmp(argv[arg_index], "difconv") == 0){
            all_data.input.test_problem = DIFCONV_3D7PT;
         }
         else if (strcmp(argv[arg_index], "vardifconv") == 0){
            all_data.input.test_problem = VARDIFCONV_3D7PT;
         }
         else if (strcmp(argv[arg_index], "mfem_laplace") == 0){
            all_data.input.test_problem = MFEM_LAPLACE;
            all_data.input.rhs_type = RHS_FROM_PROBLEM;
         }
         else if (strcmp(argv[arg_index], "mfem_elast") == 0){
            all_data.input.test_problem = MFEM_ELAST;
            all_data.input.rhs_type = RHS_FROM_PROBLEM;
         }
	 else if (strcmp(argv[arg_index], "mfem_maxwell") == 0){
            all_data.input.test_problem = MFEM_MAXWELL;
            all_data.input.rhs_type = RHS_FROM_PROBLEM;
         }
         else if (strcmp(argv[arg_index], "file") == 0){
            all_data.input.test_problem = MATRIX_FROM_FILE;
            arg_index++;
            strcpy(all_data.input.mat_file_str, argv[arg_index]);
         }
         else if (strcmp(argv[arg_index], "mfem_elast_amr") == 0){
            all_data.input.test_problem = MFEM_ELAST_AMR;
            all_data.input.rhs_type = RHS_FROM_PROBLEM;
         }
         else if (strcmp(argv[arg_index], "mfem_laplace_amr") == 0){
            all_data.input.test_problem = MFEM_LAPLACE_AMR;
            all_data.input.rhs_type = RHS_FROM_PROBLEM;
         }
      }
      else if (strcmp(argv[arg_index], "-smoother") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "j") == 0){
            all_data.input.smoother = JACOBI;
            all_data.input.smooth_interp_type = JACOBI;
            all_data.hypre.relax_type = 0;
         }
         else if (strcmp(argv[arg_index], "gs") == 0){
            all_data.input.smoother = GAUSS_SEIDEL;
            all_data.hypre.relax_type = 1;
         }
         else if (strcmp(argv[arg_index], "hybrid_jgs") == 0){
            all_data.input.smoother = HYBRID_JACOBI_GAUSS_SEIDEL;
            all_data.hypre.relax_type = 3;
         }
         else if (strcmp(argv[arg_index], "async_gs") == 0){
            all_data.input.smoother = ASYNC_GAUSS_SEIDEL;
            all_data.hypre.relax_type = 5;
         }
         else if (strcmp(argv[arg_index], "semi_async_gs") == 0){
            all_data.input.smoother = SEMI_ASYNC_GAUSS_SEIDEL;
            all_data.hypre.relax_type = 5;
         }
         else if (strcmp(argv[arg_index], "L1j") == 0){
            all_data.input.smoother = L1_JACOBI;
            all_data.input.smooth_interp_type = L1_JACOBI;
            all_data.hypre.relax_type = 18;
         }
         else if (strcmp(argv[arg_index], "L1_hybrid_jgs") == 0){
            all_data.input.smoother = L1_HYBRID_JACOBI_GAUSS_SEIDEL;
            all_data.input.smooth_interp_type = L1_HYBRID_JACOBI_GAUSS_SEIDEL;
            all_data.hypre.relax_type = 13;
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
      else if (strcmp(argv[arg_index], "-use_hypre_solver") == 0){
         all_data.input.hypre_solver_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-smooth_interp") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "j") == 0){
            all_data.input.smooth_interp_type = JACOBI;
            all_data.hypre.relax_type = 0;
         }
         else if (strcmp(argv[arg_index], "L1j") == 0){
            all_data.input.smooth_interp_type = L1_JACOBI;
            all_data.hypre.relax_type = 18;
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
      else if (strcmp(argv[arg_index], "-num_iters") == 0)
      {
         arg_index++;
         max_num_iters = start_num_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-max_num_iters") == 0)
      {
         arg_index++;
         max_num_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-start_num_iters") == 0)
      {
         arg_index++;
         start_num_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-incr_num_iters") == 0)
      {
         arg_index++;
         incr_num_iters = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-num_func") == 0)
      {
         arg_index++;
         all_data.hypre.num_functions = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-mfem_max_amr_dofs") == 0)
      {
         arg_index++;
         all_data.mfem.max_amr_dofs = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-cheby_eig_max_iters") == 0){
         arg_index++;
         all_data.input.cheby_eig_max_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-cheby_eig_tol") == 0){
         arg_index++;
         all_data.input.cheby_eig_tol = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-warmup") == 0){
         warmup = 1;
      }
      else if (strcmp(argv[arg_index], "-delay_one") == 0){
         arg_index++;
         all_data.input.delay_usec = atoi(argv[arg_index]);
         all_data.input.delay_type = DELAY_ONE;
      }
      else if (strcmp(argv[arg_index], "-delay_some") == 0){
         arg_index++;
         all_data.input.delay_usec = atoi(argv[arg_index]);
         arg_index++;
         all_data.input.delay_frac = atof(argv[arg_index]);
         all_data.input.delay_type = DELAY_SOME;
      }
      else if (strcmp(argv[arg_index], "-delay_all") == 0){
         arg_index++;
         all_data.input.delay_usec = atoi(argv[arg_index]);
         all_data.input.delay_type = DELAY_ALL;
         all_data.input.delay_frac = 1.0;
      }
      else if (strcmp(argv[arg_index], "-fail_one") == 0){
         arg_index++;
         all_data.input.delay_usec = atoi(argv[arg_index]);
         arg_index++;
         all_data.input.fail_iter = atoi(argv[arg_index]);
         all_data.input.delay_type = FAIL_ONE;
      }
      else if (strcmp(argv[arg_index], "-no_construct_R") == 0){
         all_data.input.construct_R_flag = 0;
      }
      else if (strcmp(argv[arg_index], "-background_program") == 0){
         background_program = 1; 
      }
      else if (strcmp(argv[arg_index], "-no_omp_parfor") == 0){
         all_data.input.omp_parfor_flag = 0;
      }
      else if (strcmp(argv[arg_index], "-cheby_eig") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "power") == 0){
            all_data.input.cheby_eig_type = CHEBY_EIG_POWER;
         }
         else if (strcmp(argv[arg_index], "hypre_lobpcg") == 0){
            all_data.input.cheby_eig_type = CHEBY_EIG_HYPRE_LOBPCG;
         }
         else if (strcmp(argv[arg_index], "slepc") == 0){
            all_data.input.cheby_eig_type = CHEBY_EIG_SLEPC;
         }
      }
      else if (strcmp(argv[arg_index], "-simple_jacobi") == 0){
         all_data.input.simple_jacobi_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-hypre_rlx") == 0){
         arg_index++;
         all_data.hypre.relax_type = atoi(argv[arg_index]);
      }

      arg_index++;
   }

   if (background_program == 1){
      omp_set_num_threads(all_data.input.num_threads);
      while(1){
         #pragma omp parallel
         {
            RandDouble(-1.0, 1.0);
         }
      }
      return 0;
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

   if (all_data.input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX ||
       all_data.input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX ||
       all_data.input.solver == BPX ||
       all_data.input.solver == PAR_BPX){
      if (all_data.input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL) {
         all_data.hypre.relax_type = 0;
      }
      else if (all_data.input.smoother == L1_HYBRID_JACOBI_GAUSS_SEIDEL) {
         all_data.hypre.relax_type = 18;
      }
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

   //srand(time(NULL));
   if (warmup == 1){
      num_runs++;
   }
   for (int num_iters = start_num_iters; num_iters <= max_num_iters; num_iters += incr_num_iters){   
      all_data.input.num_cycles = num_iters;
      for (int run = 1; run <= num_runs; run++){
         if (all_data.input.hypre_solver_flag == 1){
            start = omp_get_wtime();
            HYPRE_Solver pcg_solver;
            HYPRE_BoomerAMGSetMaxIter(all_data.hypre.solver, 1);
            HYPRE_BoomerAMGSetPrintLevel(all_data.hypre.solver, 0);
            HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
            HYPRE_PCGSetPrintLevel(pcg_solver, 0);
            HYPRE_PCGSetTwoNorm(pcg_solver, 1);
            HYPRE_PCGSetMaxIter(pcg_solver, all_data.input.num_cycles);
            HYPRE_PCGSetTol(pcg_solver, all_data.input.tol);
            HYPRE_PCGSetPrintLevel(pcg_solver, 0);
            HYPRE_PCGSetPrecond(pcg_solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                all_data.hypre.solver);
            start = omp_get_wtime();
            HYPRE_ParCSRPCGSetup(pcg_solver, all_data.hypre.parcsr_A, all_data.hypre.par_b, all_data.hypre.par_x);
            all_data.output.setup_wtime += omp_get_wtime() - start;
            start = omp_get_wtime();
            HYPRE_ParCSRPCGSolve(pcg_solver, all_data.hypre.parcsr_A, all_data.hypre.par_b, all_data.hypre.par_x);
            all_data.output.solve_wtime = omp_get_wtime() - start;
            int num_iterations;
            double final_res_norm;
            HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
            HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
            printf("Hypre PCG: %d %e %e\n", num_iterations, final_res_norm, all_data.output.solve_wtime);
         }
         else {
            InitSolve(&all_data);
            if (all_data.input.solver == EXPLICIT_EXTENDED_SYSTEM_BPX ||
                all_data.input.solver == IMPLICIT_EXTENDED_SYSTEM_BPX){
               SMEM_ExtendedSystemSolve(&all_data);
            }
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
   }

  // HYPRE_BoomerAMGDestroy(solver);
  // HYPRE_IJMatrixDestroy(A);
  // HYPRE_IJVectorDestroy(b);
  // HYPRE_IJVectorDestroy(x);
   
   MPI_Finalize();
   return 0;
}
