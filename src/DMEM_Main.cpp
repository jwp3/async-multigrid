#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Setup.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_Add.hpp"
#include "DMEM_Mult.hpp"

int main (int argc, char *argv[])
{
   int my_id, num_procs;
   DMEM_AllData dmem_all_data;

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
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   hypre_MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   HYPRE_Init(argc, argv);


   /* Hypre parameters */
   dmem_all_data.hypre.max_levels = 25;
   dmem_all_data.hypre.solver_id = 0;
   dmem_all_data.hypre.agg_num_levels = 0;
   dmem_all_data.hypre.coarsen_type = 9; /* for coarsening independent of the number of processes, use 7 or 9 */
   dmem_all_data.hypre.interp_type = 6;
   dmem_all_data.hypre.print_level = 0;
   dmem_all_data.hypre.solve_flag = 0;
   dmem_all_data.hypre.strong_threshold = .5;
   dmem_all_data.hypre.multadd_trunc_factor = 0.0;
   dmem_all_data.hypre.start_smooth_level = 0;
   dmem_all_data.hypre.num_functions = 1;
   dmem_all_data.hypre.P_max_elmts = 3;
   dmem_all_data.hypre.add_P_max_elmts = 0;
   dmem_all_data.hypre.add_trunc_factor = 0.0;

   int n = 40;
   double c = 1.0, a = 1.0;
   dmem_all_data.matrix.nx = n;
   dmem_all_data.matrix.ny = n;
   dmem_all_data.matrix.nz = n;
   dmem_all_data.matrix.difconv_ax = a;
   dmem_all_data.matrix.difconv_ay = a;
   dmem_all_data.matrix.difconv_az = a;
   dmem_all_data.matrix.difconv_cx = c;
   dmem_all_data.matrix.difconv_cy = c;
   dmem_all_data.matrix.difconv_cz = c;
   dmem_all_data.matrix.difconv_atype = 0;
   dmem_all_data.matrix.vardifconv_eps = 1.0;

   /* mfem parameters */
   dmem_all_data.mfem.ref_levels = 1;
   dmem_all_data.mfem.par_ref_levels = 0;
   dmem_all_data.mfem.order = 1;
   dmem_all_data.mfem.max_amr_iters = 1000;
   dmem_all_data.mfem.max_amr_dofs = 50000;
   strcpy(dmem_all_data.mfem.mesh_file, "./mfem_quartz/mfem-3.4/data/beam-tet.mesh");

   dmem_all_data.input.test_problem = LAPLACE_3D27PT;
   dmem_all_data.input.tol = 1e-10;
   dmem_all_data.input.inner_tol = .001;
   dmem_all_data.input.async_flag = 0;
   dmem_all_data.input.async_smoother_flag = 0;
   dmem_all_data.input.global_conv_flag = 0;
   dmem_all_data.input.assign_procs_type = ASSIGN_PROCS_BALANCED_WORK;
   dmem_all_data.input.assign_procs_scalar = .5;
   dmem_all_data.input.converge_test_type = GLOBAL_CONVERGE;
   dmem_all_data.input.res_compute_type = LOCAL_RES;
   dmem_all_data.input.num_pre_smooth_sweeps = 1;
   dmem_all_data.input.num_post_smooth_sweeps = 1;
   dmem_all_data.input.num_fine_smooth_sweeps = 1;
   dmem_all_data.input.num_coarse_smooth_sweeps = 1;
   dmem_all_data.input.oneline_output_flag = 0;
   dmem_all_data.input.num_threads = 1;
   dmem_all_data.input.print_output_flag = 1;
   dmem_all_data.input.smooth_weight = 1.0;
   dmem_all_data.input.smoother = JACOBI;
   dmem_all_data.input.smooth_interp_type = JACOBI;
   dmem_all_data.input.solver = MULT;
   dmem_all_data.input.hypre_test_error_flag = 0;
   dmem_all_data.input.mfem_test_error_flag = 0;
   dmem_all_data.input.mfem_solve_print_flag = 0;
   dmem_all_data.input.print_level_stats_flag = 0;
   dmem_all_data.input.print_reshist_flag = 0;
   dmem_all_data.input.read_type = READ_SOL;
   dmem_all_data.input.num_cycles = 1000;
   dmem_all_data.input.num_inner_cycles = 5;
   dmem_all_data.input.increment_cycle = 1;
   dmem_all_data.input.start_cycle = 1;
   dmem_all_data.input.coarsest_mult_level = 0;
   dmem_all_data.input.check_res_flag = 1;
   dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_ALL_LEVELS;
   dmem_all_data.input.max_inflight = 1;
   dmem_all_data.input.rhs_type = RHS_RAND;
   dmem_all_data.input.init_guess_type = INITGUESS_RAND;
   dmem_all_data.input.afacj_level = 0;
   dmem_all_data.input.num_interpolants = NUMLEVELS_INTERPOLANTS;
   dmem_all_data.input.res_update_type = RES_RECOMPUTE;
   dmem_all_data.input.sps_probability_type = SPS_PROBABILITY_EXPONENTIAL;
   dmem_all_data.input.sps_alpha = 1.0;
   dmem_all_data.input.simple_jacobi_flag = 0;
   dmem_all_data.input.async_comm_save_divisor = 1;
   dmem_all_data.input.optimal_jacobi_weight_flag = 0;
   dmem_all_data.input.eig_CG_max_iter = 20;

//#ifdef HYPRE_USING_UNIFIED_MEMORY
//   dmem_all_data.input.hypre_memory = HYPRE_MEMORY_SHARED;
//#else
   dmem_all_data.input.hypre_memory = HYPRE_MEMORY_SHARED;
//#endif

   /* Parse command line */
   int arg_index = 0;
   int print_usage = 0;
   int coarsest_mult_level = 1;

   int num_solvers = 0;
   vector<int> solvers;

   while (arg_index < argc){
      if (strcmp(argv[arg_index], "-n") == 0){
         arg_index++;
	 n = atoi(argv[arg_index]);
         dmem_all_data.mfem.max_amr_dofs = n;
	 dmem_all_data.matrix.nx = n;
         dmem_all_data.matrix.ny = n;
         dmem_all_data.matrix.nz = n;
      }
      else if (strcmp(argv[arg_index], "-nx") == 0){
         arg_index++;
         dmem_all_data.matrix.nx = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-ny") == 0){
         arg_index++;
         dmem_all_data.matrix.ny = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-nz") == 0){
         arg_index++;
         dmem_all_data.matrix.nz = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-c") == 0){
         arg_index++;
         c = atof(argv[arg_index]);
         dmem_all_data.matrix.difconv_cx = c;
         dmem_all_data.matrix.difconv_cy = c;
         dmem_all_data.matrix.difconv_cz = c;
      }
      else if (strcmp(argv[arg_index], "-cx") == 0){
         arg_index++;
         dmem_all_data.matrix.difconv_cx = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-cy") == 0){
         arg_index++;
         dmem_all_data.matrix.difconv_cy = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-cz") == 0){
         arg_index++;
         dmem_all_data.matrix.difconv_cz = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-a") == 0){
         arg_index++;
         a = atof(argv[arg_index]);
         dmem_all_data.matrix.difconv_ax = a;
         dmem_all_data.matrix.difconv_ay = a;
         dmem_all_data.matrix.difconv_az = a;
      }
      else if (strcmp(argv[arg_index], "-ax") == 0){
         arg_index++;
         dmem_all_data.matrix.difconv_ax = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-ay") == 0){
         arg_index++;
         dmem_all_data.matrix.difconv_ay = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-az") == 0){
         arg_index++;
         dmem_all_data.matrix.difconv_az = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-atype") == 0){
         arg_index++;
         dmem_all_data.matrix.difconv_atype = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-problem") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "27pt") == 0){
            dmem_all_data.input.test_problem = LAPLACE_3D27PT;
         }
         else if (strcmp(argv[arg_index], "7pt") == 0){
            dmem_all_data.input.test_problem = LAPLACE_3D7PT;
         }
         else if (strcmp(argv[arg_index], "difconv") == 0){
            dmem_all_data.input.test_problem = DIFCONV_3D7PT;
         }
	 else if (strcmp(argv[arg_index], "vardifconv") == 0){
            dmem_all_data.input.test_problem = VARDIFCONV_3D7PT;
         }
         else if (strcmp(argv[arg_index], "mfem_elast") == 0){
            dmem_all_data.input.test_problem = MFEM_ELAST;
         }
         else if (strcmp(argv[arg_index], "mfem_elast_amr") == 0){
            dmem_all_data.input.test_problem = MFEM_ELAST_AMR;
         }
         else if (strcmp(argv[arg_index], "file") == 0){
            dmem_all_data.input.test_problem = MATRIX_FROM_FILE;
            arg_index++;
            strcpy(dmem_all_data.input.mat_file_str, argv[arg_index]);
         }
      }
      else if (strcmp(argv[arg_index], "-vardifconv_eps") == 0){
         arg_index++;
         dmem_all_data.matrix.vardifconv_eps = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-smoother") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "j") == 0){
            dmem_all_data.input.smoother = JACOBI;
            dmem_all_data.input.smooth_interp_type = JACOBI;
         }
         else if (strcmp(argv[arg_index], "hjgs") == 0){
            dmem_all_data.input.smoother = HYBRID_JACOBI_GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "L1j") == 0){
            dmem_all_data.input.smoother = L1_JACOBI;
            dmem_all_data.input.smooth_interp_type = L1_JACOBI;
         }
         else if (strcmp(argv[arg_index], "async_j") == 0){
            dmem_all_data.input.smoother = ASYNC_JACOBI;
            dmem_all_data.input.smooth_interp_type = JACOBI;
            dmem_all_data.input.async_smoother_flag = 1;
         }
         else if (strcmp(argv[arg_index], "async_L1j") == 0){
            dmem_all_data.input.smoother = ASYNC_L1_JACOBI;
            dmem_all_data.input.smooth_interp_type = L1_JACOBI;
            dmem_all_data.input.async_smoother_flag = 1;
         }
         else if (strcmp(argv[arg_index], "async_sps") == 0){
            dmem_all_data.input.smoother = ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL;
            dmem_all_data.input.smooth_interp_type = JACOBI;
            dmem_all_data.input.async_smoother_flag = 1;
         }
         else if (strcmp(argv[arg_index], "async_hjgs") == 0){
            dmem_all_data.input.smoother = ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL;
            dmem_all_data.input.smooth_interp_type = JACOBI;
            dmem_all_data.input.async_smoother_flag = 1;
         }
      }
      else if (strcmp(argv[arg_index], "-solver") == 0){
         arg_index++;
         dmem_all_data.input.solver = atoi(argv[arg_index]);
         if (strcmp(argv[arg_index], "mult") == 0){
            dmem_all_data.input.solver = MULT;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
         else if (strcmp(argv[arg_index], "multadd") == 0){
            dmem_all_data.input.solver = MULTADD;
            dmem_all_data.input.async_flag = 0;
         }
         else if (strcmp(argv[arg_index], "async_multadd") == 0){
            dmem_all_data.input.solver = MULTADD;
            dmem_all_data.input.async_flag = 1;
         }
         else if (strcmp(argv[arg_index], "afacj") == 0){
            dmem_all_data.input.solver = MULTADD;
            dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_MY_GRID;
            dmem_all_data.input.async_flag = 0;
         }
         else if (strcmp(argv[arg_index], "async_afacj") == 0){
            dmem_all_data.input.solver = MULTADD;
            dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_MY_GRID;
            dmem_all_data.input.async_flag = 1;
         }
         else if (strcmp(argv[arg_index], "afacx") == 0){
            dmem_all_data.input.solver = AFACX;
            dmem_all_data.input.async_flag = 0;
         }
         else if (strcmp(argv[arg_index], "mult_multadd") == 0){
            dmem_all_data.input.solver = MULT_MULTADD;
         }
         else if (strcmp(argv[arg_index], "sync_multadd") == 0){
            dmem_all_data.input.solver = SYNC_MULTADD;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
         else if (strcmp(argv[arg_index], "sync_afacx") == 0){
            dmem_all_data.input.solver = SYNC_AFACX;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
        // solvers.push_back(dmem_all_data.input.solver);
        // num_solvers++;
      }
      else if (strcmp(argv[arg_index], "-simple_jacobi") == 0){
         dmem_all_data.input.simple_jacobi_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-multadd_smooth_interp_level") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "all") == 0){
            dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_ALL_LEVELS;
         }
         else if (strcmp(argv[arg_index], "grid") == 0){
            dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_MY_GRID;
         }
      }
      else if (strcmp(argv[arg_index], "-rhs") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "zeros") == 0){
            dmem_all_data.input.rhs_type = RHS_ZEROS;
         }
         else if (strcmp(argv[arg_index], "ones") == 0){
            dmem_all_data.input.rhs_type = RHS_ONES;
         }
         else if (strcmp(argv[arg_index], "rand") == 0){
            dmem_all_data.input.rhs_type = RHS_RAND;
         }
         else if (strcmp(argv[arg_index], "from_problem") == 0){
            dmem_all_data.input.rhs_type = RHS_FROM_PROBLEM;
         }
      }
      else if (strcmp(argv[arg_index], "-init_guess") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "zeros") == 0){
            dmem_all_data.input.init_guess_type = INITGUESS_ZEROS;
         }
         else if (strcmp(argv[arg_index], "ones") == 0){
            dmem_all_data.input.init_guess_type = INITGUESS_ONES;
         }
         else if (strcmp(argv[arg_index], "rand") == 0){
            dmem_all_data.input.init_guess_type = INITGUESS_RAND;
         }
      }
      else if (strcmp(argv[arg_index], "-assign_procs") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "balanced") == 0){
            dmem_all_data.input.assign_procs_type = ASSIGN_PROCS_BALANCED_WORK;
         }
         else if (strcmp(argv[arg_index], "scalar") == 0){
            dmem_all_data.input.assign_procs_type = ASSIGN_PROCS_SCALAR;
         }
      }
      else if (strcmp(argv[arg_index], "-assign_procs_scalar") == 0){
         arg_index++;
         dmem_all_data.input.assign_procs_scalar = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-async") == 0){
         dmem_all_data.input.async_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-max_inflight") == 0){
         arg_index++;
         dmem_all_data.input.max_inflight = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-coarsest_mult_level") == 0){
         arg_index++;
         coarsest_mult_level = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-afacj_level") == 0){
         arg_index++;
         dmem_all_data.input.afacj_level = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-smooth_weight") == 0){
         arg_index++;
         dmem_all_data.input.smooth_weight = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-sps_alpha") == 0){
         arg_index++;
         dmem_all_data.input.sps_alpha = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-th") == 0){
         arg_index++;
         dmem_all_data.hypre.strong_threshold = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_cycles") == 0){
         arg_index++;
         dmem_all_data.input.num_cycles = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_inner_cycles") == 0){
         arg_index++;
         dmem_all_data.input.num_inner_cycles = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-start_cycle") == 0){
         arg_index++;
         dmem_all_data.input.start_cycle = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-incr_cycle") == 0){
         arg_index++;
         dmem_all_data.input.increment_cycle = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-tol") == 0){
         arg_index++;
         dmem_all_data.input.tol = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-inner_tol") == 0){
         arg_index++;
         dmem_all_data.input.inner_tol = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_smooth_sweeps") == 0){
         arg_index++;
         dmem_all_data.input.num_pre_smooth_sweeps = atoi(argv[arg_index]);
         dmem_all_data.input.num_post_smooth_sweeps = atoi(argv[arg_index]);
         dmem_all_data.input.num_fine_smooth_sweeps = atoi(argv[arg_index]);
         dmem_all_data.input.num_coarse_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_pre_smooth_sweeps") == 0){
         arg_index++;
         dmem_all_data.input.num_pre_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_post_smooth_sweeps") == 0){
         arg_index++;
         dmem_all_data.input.num_post_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_fine_smooth_sweeps") == 0){
         arg_index++;
         dmem_all_data.input.num_fine_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_coarse_smooth_sweeps") == 0){
         arg_index++;
         dmem_all_data.input.num_coarse_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mxl") == 0){
         arg_index++;
         dmem_all_data.hypre.max_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-agg_nl") == 0){
         arg_index++;
         dmem_all_data.hypre.agg_num_levels = atoi(argv[arg_index]);
      }
      else if ( strcmp(argv[arg_index], "-Pmx") == 0 ){
         arg_index++;
         dmem_all_data.hypre.P_max_elmts = atoi(argv[arg_index]);
      }
      else if ( strcmp(argv[arg_index], "-add_Pmx") == 0 ){
         arg_index++;
         dmem_all_data.hypre.add_P_max_elmts = atoi(argv[arg_index]);
      }
      else if ( strcmp(argv[arg_index], "-add_tr") == 0 ){
         arg_index++;
         dmem_all_data.hypre.add_trunc_factor = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-coarsen_type") == 0){
         arg_index++;
         dmem_all_data.hypre.coarsen_type = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-interp_type") == 0){
         arg_index++;
         dmem_all_data.hypre.interp_type = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_ref_levels") == 0){
         arg_index++;
         dmem_all_data.mfem.ref_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_par_ref_levels") == 0){
         arg_index++;
         dmem_all_data.mfem.par_ref_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_max_amr_iters") == 0){
         arg_index++;
         dmem_all_data.mfem.max_amr_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_max_amr_dofs") == 0){
         arg_index++;
         dmem_all_data.mfem.max_amr_dofs = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_order") == 0){
         arg_index++;
         dmem_all_data.mfem.order = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_mesh_file") == 0){
         arg_index++;
         strcpy(dmem_all_data.mfem.mesh_file, argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-converge_test_type") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "local") == 0){
            dmem_all_data.input.converge_test_type = LOCAL_CONVERGE;
         }
         else if (strcmp(argv[arg_index], "global") == 0){
            dmem_all_data.input.converge_test_type = GLOBAL_CONVERGE;
         }
      }
      else if (strcmp(argv[arg_index], "-res_compute_type") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "local") == 0){
            dmem_all_data.input.res_compute_type = LOCAL_RES;
         }
         else if (strcmp(argv[arg_index], "global") == 0){
            dmem_all_data.input.res_compute_type = GLOBAL_RES;
         }
      }
      else if (strcmp(argv[arg_index], "-res_update_type") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "accum") == 0){
            dmem_all_data.input.res_update_type = RES_ACCUMULATE;
         }
         else if (strcmp(argv[arg_index], "recomp") == 0){
            dmem_all_data.input.res_update_type = RES_RECOMPUTE;
         }
      }
      else if (strcmp(argv[arg_index], "-num_interps") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "one") == 0){
            dmem_all_data.input.num_interpolants = ONE_INTERPOLANT;
         }
         else if (strcmp(argv[arg_index], "num_levels") == 0){
            dmem_all_data.input.num_interpolants = NUMLEVELS_INTERPOLANTS;
         }
      }
      else if (strcmp(argv[arg_index], "-num_runs") == 0){
         arg_index++;
         num_runs = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-async_comm_save") == 0){
         arg_index++;
         dmem_all_data.input.async_comm_save_divisor = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-read_type") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "sol") == 0){
            dmem_all_data.input.read_type = READ_SOL;
         }
         else if (strcmp(argv[arg_index], "res") == 0){
            dmem_all_data.input.read_type = READ_RES;
         }
      }
      else if (strcmp(argv[arg_index], "-no_output") == 0){
         dmem_all_data.input.print_output_flag = 0;
      }
      else if (strcmp(argv[arg_index], "-oneline_output") == 0){
         dmem_all_data.input.oneline_output_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_output_level") == 0){
         arg_index++;
         dmem_all_data.input.print_output_level = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-print_reshist") == 0){
         dmem_all_data.input.print_reshist_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_hypre") == 0){
         dmem_all_data.hypre.print_level = 3;
      }
      else if (strcmp(argv[arg_index], "-print_level_stats") == 0){
         dmem_all_data.input.print_level_stats_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-hypre_test_error") == 0){
         dmem_all_data.input.hypre_test_error_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-mfem_test_error") == 0){
         dmem_all_data.input.mfem_test_error_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-optimal_jacobi_weight") == 0){
         dmem_all_data.input.optimal_jacobi_weight_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-eig_cg_max_iter") == 0){
         arg_index++;
         dmem_all_data.input.eig_CG_max_iter = atoi(argv[arg_index]);
      }
      arg_index++;
   }

   dmem_all_data.input.async_comm_save_divisor = max(1, dmem_all_data.input.async_comm_save_divisor);

   if (dmem_all_data.input.solver == MULT_MULTADD){
      dmem_all_data.input.coarsest_mult_level = coarsest_mult_level;
   } 
   else {
      dmem_all_data.input.coarsest_mult_level = 0;
   }

   if (num_solvers == 0){
      solvers.push_back(dmem_all_data.input.solver);
      num_solvers = 1;
   }

   if ((dmem_all_data.input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL || dmem_all_data.input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL || dmem_all_data.input.smoother == ASYNC_JACOBI) &&
       dmem_all_data.input.async_flag == 0){
      dmem_all_data.input.smoother = JACOBI;
   }
   else if ((dmem_all_data.input.solver == SYNC_AFACX || dmem_all_data.input.solver == SYNC_MULTADD || dmem_all_data.input.solver == MULT) &&
            dmem_all_data.input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
      dmem_all_data.input.smoother = JACOBI;
   }

   if ((dmem_all_data.input.smoother == ASYNC_L1_JACOBI) &&
       dmem_all_data.input.async_flag == 0){
      dmem_all_data.input.smoother = L1_JACOBI;
   }
   
   if (dmem_all_data.input.rhs_type == RHS_FROM_PROBLEM){
      if (dmem_all_data.input.test_problem != MFEM_ELAST && 
          dmem_all_data.input.test_problem != MFEM_ELAST_AMR){
         dmem_all_data.input.rhs_type = RHS_RAND;
      }
   }

  // if (dmem_all_data.input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
  //    dmem_all_data.input.res_update_type = RES_ACCUMULATE;
  // }

   dmem_all_data.grid.my_grid = 0;
   
  // srand(0);
  // mkl_set_num_threads(1);

   start = omp_get_wtime(); 
   DMEM_Setup(&dmem_all_data);
   dmem_all_data.output.setup_wtime = omp_get_wtime() - start;

   for (int s = 0; s < num_solvers; s++){
      dmem_all_data.input.solver = solvers[s];
      for (int run = 0; run < num_runs; run++){
         DMEM_ResetData(&dmem_all_data);

         if (dmem_all_data.input.oneline_output_flag == 0 && my_id == 0){
            printf("\nSOLVER: ");
            if (dmem_all_data.input.async_flag == 1){
               printf("asynchronous ");
            }
         }
         if (dmem_all_data.input.solver == MULT){
            if (dmem_all_data.input.oneline_output_flag == 0 && my_id == 0){
               printf("classical multiplicative\n\n\n");
            }
            DMEM_Mult(&dmem_all_data);
         }
         else if (dmem_all_data.input.solver == MULT_MULTADD){
            if (dmem_all_data.input.oneline_output_flag == 0 && my_id == 0){
               printf("classical multiplicative with multadd as coarse grid solver\n\n\n");
            }
            DMEM_Mult(&dmem_all_data);
         }
         else if (dmem_all_data.input.solver == SYNC_MULTADD || dmem_all_data.input.solver == SYNC_AFACX){
            if (dmem_all_data.input.solver == SYNC_AFACX){
               if (dmem_all_data.input.oneline_output_flag == 0 && my_id == 0){
                  printf("synchronous AFACx\n\n\n");
               }
            }
            else {
               if (dmem_all_data.input.oneline_output_flag == 0 && my_id == 0){
                  printf("synchronous multadd\n\n\n");
               }
            }
            DMEM_SyncAdd(&dmem_all_data);
         }
         else {
            if (dmem_all_data.input.oneline_output_flag == 0 && my_id == 0){
               if (dmem_all_data.input.solver == BPX){
                  printf("BPX\n\n\n");
               }
               else if (dmem_all_data.input.solver == AFACX){
                  printf("AFACx\n\n\n");
               }
               else {
                  if (dmem_all_data.input.multadd_smooth_interp_level_type == SMOOTH_INTERP_MY_GRID){
                     printf("AFACj\n\n\n");
                  }
                  else {
                     printf("multadd\n\n\n");
                  }
               }
            }
            DMEM_Add(&dmem_all_data);
         }
         DMEM_PrintOutput(&dmem_all_data);
      }
   }

  // HYPRE_BoomerAMGSolve(dmem_all_data.hypre.solver,
  //                      dmem_all_data.matrix.A_fine,
  //                      dmem_all_data.vector_fine.f,
  //                      dmem_all_data.vector_fine.u);

  // if (dmem_all_data.input.solver == MULTADD){
  //    HYPRE_Real *e_local_data;
  //    hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data.hypre.solver_gridk;
  //    hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data);
  //    hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
  //    hypre_ParVector **U_array = hypre_ParAMGDataUArray(amg_data);
  //    hypre_ParVector **F_array = hypre_ParAMGDataFArray(amg_data);
  //    HYPRE_Real *u_local_data = hypre_VectorData(hypre_ParVectorLocalVector(U_array[0]));
  //    HYPRE_Real *f_local_data = hypre_VectorData(hypre_ParVectorLocalVector(F_array[0]));
  //    HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data.vector_gridk.x));
  //    HYPRE_Real *r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data.vector_gridk.r));
  //    HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));

  //    HYPRE_Int my_grid = dmem_all_data.grid.my_grid;
  //    for (HYPRE_Int p = 0; p < num_procs; p++){
  //       if (my_id == p){
  //          for (HYPRE_Int i = 0; i < num_rows; i++){
  //            // printf("%d %e\n", my_grid, u_local_data[i]);
  //          }
  //       }
  //       MPI_Barrier(MPI_COMM_WORLD);
  //    }

  //    if (my_id == 0) printf("\n");
  //    MPI_Barrier(MPI_COMM_WORLD);

  //    amg_data = (hypre_ParAMGData *)dmem_all_data.hypre.solver;
  //    A_array = hypre_ParAMGDataAArray(amg_data);
  //    r_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data.vector_fine.r));
  //    x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data.vector_fine.x));
  //    e_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data.vector_fine.e));
  //    num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[0]));

  //    int my_grid = dmem_all_data.grid.my_grid;
  //    for (HYPRE_Int p = 0; p < num_procs; p++){
  //       if (my_id == p){
  //          for (HYPRE_Int i = 0; i < num_rows; i++){
  //             printf("%d %e %e\n", my_grid, x_local_data[i], r_local_data[i]);
  //          }
  //       }
  //       MPI_Barrier(MPI_COMM_WORLD);
  //    }
  // }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return 0;
}
