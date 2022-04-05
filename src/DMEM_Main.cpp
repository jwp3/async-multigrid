#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Setup.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_Add.hpp"
#include "DMEM_Mult.hpp"
#include "DMEM_Eig.hpp"

HYPRE_Int vecop_machine;

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
   dmem_all_data.hypre.P_max_elmts = 0;
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
   dmem_all_data.input.outer_tol = 1e-8;
   dmem_all_data.input.outer_max_iter = 500;
   dmem_all_data.input.async_flag = 0;
   dmem_all_data.input.async_smoother_flag = 0;
   dmem_all_data.input.global_conv_flag = 0;
   dmem_all_data.input.assign_procs_type = ASSIGN_PROCS_BALANCED_WORK;
   dmem_all_data.input.assign_procs_scalar = .5;
   dmem_all_data.input.converge_test_type = GLOBAL_CONVERGE;
   dmem_all_data.input.res_compute_type = LOCAL_RES;
   dmem_all_data.input.num_add_smooth_sweeps = 2;
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
   dmem_all_data.input.outer_solver = NO_OUTER_SOLVER;
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
   dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_MULTADD;
   dmem_all_data.input.max_inflight = 1;
   dmem_all_data.input.rhs_type = RHS_RAND;
   dmem_all_data.input.init_guess_type = INITGUESS_ZEROS;
   dmem_all_data.input.afacj_level = 1;
   dmem_all_data.input.num_interpolants = NUMLEVELS_INTERPOLANTS;
   dmem_all_data.input.res_update_type = RES_RECOMPUTE;
   dmem_all_data.input.sps_probability_type = SPS_PROBABILITY_EXPONENTIAL;
   dmem_all_data.input.sps_alpha = 1.0;
   dmem_all_data.input.sps_min_prob = 0;
   dmem_all_data.input.simple_jacobi_flag = -1;
   dmem_all_data.input.async_comm_save_divisor = 1;
   dmem_all_data.input.optimal_jacobi_weight_flag = 1;
   dmem_all_data.input.eig_CG_max_iters = 20;
   dmem_all_data.input.only_setup_flag = 0;
   dmem_all_data.input.only_build_matrix_flag = 0;
   dmem_all_data.input.include_disconnected_points_flag = 0;
   dmem_all_data.input.eig_power_max_iters = 20;
   dmem_all_data.input.accel_type = NO_ACCEL;
   dmem_all_data.input.print_matrix_flag = 0;
   dmem_all_data.input.par_file_flag = 0;
   dmem_all_data.input.imbal = 1.0;
   dmem_all_data.input.async_type = FULL_ASYNC;
   dmem_all_data.input.eig_power_MG_max_iters  = 1;
   dmem_all_data.input.eig_shift = 0.0;
   dmem_all_data.input.b_eig_shift = 0.0;
   dmem_all_data.input.a_eig_shift = 0.0;
   dmem_all_data.input.delay_usec = 0;
   dmem_all_data.input.delay_flag = 0;
   dmem_all_data.input.delay_id = num_procs-1;
   dmem_all_data.input.cheby_grid = 0;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_UNIFIED_MEMORY)
   hypre_SetExecPolicy(HYPRE_EXEC_HOST);
   dmem_all_data.input.hypre_memory = HYPRE_MEMORY_HOST;
   vecop_machine = HYPRE_MEMORY_HOST;
#else
   dmem_all_data.input.hypre_memory = HYPRE_MEMORY_HOST;
   vecop_machine = HYPRE_MEMORY_HOST;
#endif

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
         if (strcmp(argv[arg_index], "5pt") == 0){
            dmem_all_data.input.test_problem = LAPLACE_2D5PT;
         }
         else if (strcmp(argv[arg_index], "27pt") == 0){
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
            dmem_all_data.input.smooth_interp_type = JACOBI;
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
            dmem_all_data.input.smoother = ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI;
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
            dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_AFACJ;
            dmem_all_data.input.async_flag = 0;
         }
         else if (strcmp(argv[arg_index], "async_afacj") == 0){
            dmem_all_data.input.solver = MULTADD;
            dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_AFACJ;
            dmem_all_data.input.async_flag = 1;
         }
        // else if (strcmp(argv[arg_index], "afacx") == 0){
        //    dmem_all_data.input.solver = AFACX;
        //    dmem_all_data.input.async_flag = 0;
        // }
         else if (strcmp(argv[arg_index], "afacx") == 0){
            dmem_all_data.input.solver = MULTADD;
            dmem_all_data.input.multadd_smooth_interp_level_type = SMOOTH_INTERP_AFACX;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.simple_jacobi_flag = 1;
         }
         else if (strcmp(argv[arg_index], "mult_multadd") == 0){
            dmem_all_data.input.solver = MULT_MULTADD;
         }
         else if (strcmp(argv[arg_index], "sync_multadd") == 0){
            dmem_all_data.input.solver = SYNC_MULTADD;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
         else if (strcmp(argv[arg_index], "sync_afacj") == 0){
            dmem_all_data.input.solver = SYNC_AFACJ;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
         else if (strcmp(argv[arg_index], "sync_afacx") == 0){
            dmem_all_data.input.solver = SYNC_AFACX;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
         else if (strcmp(argv[arg_index], "sync_bpx") == 0){
            dmem_all_data.input.solver = SYNC_BPX;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
         else if (strcmp(argv[arg_index], "boomeramg") == 0){
            dmem_all_data.input.solver = BOOMERAMG;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
         else if (strcmp(argv[arg_index], "boomeramg_multadd") == 0){
            dmem_all_data.input.solver = BOOMERAMG_MULTADD;
            dmem_all_data.input.async_flag = 0;
            dmem_all_data.input.async_smoother_flag = 0;
         }
        // solvers.push_back(dmem_all_data.input.solver);
        // num_solvers++;
      }
      else if (strcmp(argv[arg_index], "-hypre_pcg") == 0){
         dmem_all_data.input.outer_solver = HYPRE_PCG;
      }
      else if (strcmp(argv[arg_index], "-simple_jacobi") == 0){
         dmem_all_data.input.simple_jacobi_flag = 1;
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
      else if (strcmp(argv[arg_index], "-hypre_mem") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "host") == 0){
            dmem_all_data.input.hypre_memory = HYPRE_MEMORY_HOST;
         }
         else if (strcmp(argv[arg_index], "shared") == 0){
            dmem_all_data.input.hypre_memory = HYPRE_MEMORY_SHARED;
         }
      }
      else if (strcmp(argv[arg_index], "-vecop") == 0){
         arg_index++;
         if (strcmp(argv[arg_index], "host") == 0){
            vecop_machine = HYPRE_MEMORY_HOST;
         }
         else if (strcmp(argv[arg_index], "device") == 0){
            vecop_machine = HYPRE_MEMORY_SHARED;
         }
      }
      else if (strcmp(argv[arg_index], "-assign_procs_scalar") == 0){
         arg_index++;
         dmem_all_data.input.assign_procs_scalar = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-async_flag") == 0){
         arg_index++;
         dmem_all_data.input.async_flag = atoi(argv[arg_index]);
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
         dmem_all_data.input.afacj_level = max(1, atoi(argv[arg_index]));
      }
      else if (strcmp(argv[arg_index], "-smooth_weight") == 0){
         arg_index++;
         dmem_all_data.input.smooth_weight = atof(argv[arg_index]);
         dmem_all_data.input.optimal_jacobi_weight_flag = 0; 
      }
      else if (strcmp(argv[arg_index], "-sps_alpha") == 0){
         arg_index++;
         dmem_all_data.input.sps_alpha = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-sps_min_prob") == 0){
         arg_index++;
         dmem_all_data.input.sps_min_prob = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-sps_rand") == 0){
         arg_index++;
         dmem_all_data.input.sps_alpha = atof(argv[arg_index]);
         dmem_all_data.input.sps_probability_type = SPS_PROBABILITY_RANDOM;
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
         dmem_all_data.input.num_add_smooth_sweeps = atoi(argv[arg_index]);
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
      else if (strcmp(argv[arg_index], "-print_matrix") == 0){
         dmem_all_data.input.print_matrix_flag = 1;
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
      else if (strcmp(argv[arg_index], "-eig_CG_max_iters") == 0){
         arg_index++;
         dmem_all_data.input.eig_CG_max_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-eig_power_max_iters") == 0){
         arg_index++;
         dmem_all_data.input.eig_power_max_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-eig_power_MG_max_iters") == 0){
         arg_index++;
         dmem_all_data.input.eig_power_MG_max_iters = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-only_setup") == 0){
         dmem_all_data.input.only_setup_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-only_build_matrix") == 0){
         dmem_all_data.input.only_build_matrix_flag = 1;
         dmem_all_data.input.only_setup_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-include_disconnected_points") == 0){
         dmem_all_data.input.include_disconnected_points_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-cheby") == 0){
         dmem_all_data.input.accel_type = CHEBY_ACCEL;
      }
      else if (strcmp(argv[arg_index], "-richard") == 0){
         dmem_all_data.input.accel_type = RICHARD_ACCEL;
      }
      else if (strcmp(argv[arg_index], "-par_file") == 0){
         dmem_all_data.input.par_file_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-num_funcs") == 0){
         arg_index++;
         dmem_all_data.hypre.num_functions = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-imbal") == 0){
         arg_index++;
         dmem_all_data.input.imbal = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-delay") == 0){
         arg_index++;
         dmem_all_data.input.delay_usec = atoi(argv[arg_index]);
         dmem_all_data.input.delay_flag = 1;
      } 
      else if (strcmp(argv[arg_index], "-eig_shift") == 0){
         arg_index++;
         dmem_all_data.input.b_eig_shift = dmem_all_data.input.a_eig_shift = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-b_eig_shift") == 0){
         arg_index++;
         dmem_all_data.input.b_eig_shift = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-a_eig_shift") == 0){
         arg_index++;
         dmem_all_data.input.a_eig_shift = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-cheby_grid") == 0){
         arg_index++;
         dmem_all_data.input.cheby_grid = atoi(argv[arg_index]);
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

   if ((dmem_all_data.input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
        dmem_all_data.input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL || 
        dmem_all_data.input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL || 
        dmem_all_data.input.smoother == ASYNC_JACOBI) &&
       dmem_all_data.input.async_flag == 0){
      dmem_all_data.input.smoother = JACOBI;
   }
   else if ((dmem_all_data.input.solver == SYNC_AFACX ||
             dmem_all_data.input.solver == SYNC_MULTADD || 
             dmem_all_data.input.solver == MULT ||
             dmem_all_data.input.solver == SYNC_AFACJ) &&
            (dmem_all_data.input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_JACOBI ||
             dmem_all_data.input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL_GAUSS_SEIDEL)){
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

   if (dmem_all_data.input.test_problem == MFEM_ELAST_AMR){
      dmem_all_data.mfem.max_amr_dofs = dmem_all_data.matrix.nx * dmem_all_data.matrix.ny * dmem_all_data.matrix.nz;
   }

  // if (dmem_all_data.input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL){
  //    dmem_all_data.input.res_update_type = RES_ACCUMULATE;
  // }

   dmem_all_data.grid.my_grid = 0;
   
  // srand(0);
  // mkl_set_num_threads(1);

   start = MPI_Wtime(); 
   DMEM_Setup(&dmem_all_data);
   dmem_all_data.output.setup_wtime = MPI_Wtime() - start;
   if (dmem_all_data.input.only_setup_flag == 1){
      MPI_Finalize();
      return 0;
   }

   HYPRE_Real final_res_norm;
   HYPRE_Int num_iterations;

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
         if (dmem_all_data.input.outer_solver == HYPRE_PCG){
            if (my_id == 0){
               printf("PCG\n");
            }
            start = MPI_Wtime();  
            HYPRE_PCGSolve(dmem_all_data.hypre.outer_solver,
                           (HYPRE_Matrix)(dmem_all_data.matrix.A_fine),
                           (HYPRE_Vector)(dmem_all_data.vector_fine.f),
                           (HYPRE_Vector)(dmem_all_data.vector_fine.u));
            HYPRE_Real pcg_wtime = MPI_Wtime() - start;
            HYPRE_PCGGetNumIterations(dmem_all_data.hypre.outer_solver, &num_iterations);
            HYPRE_PCGGetFinalRelativeResidualNorm(dmem_all_data.hypre.outer_solver, &final_res_norm);
            if (my_id == 0){
               hypre_printf("\n");
               hypre_printf("Iterations = %d\n", num_iterations);
               hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
               hypre_printf("PCG Wall-clock Time = %e\n", pcg_wtime);
               hypre_printf("Precond Wall-clock Time = %e\n", hypre_precond_wtime);
               hypre_printf("Precond Comm Wall-clock Time = %e\n", hypre_comm_wtime);
               hypre_printf("Precond Smooth Wall-clock Time = %e\n", hypre_smooth_wtime);
               hypre_printf("Precond Restrict Wall-clock Time = %e\n", hypre_restrict_wtime);
               hypre_printf("Precond Prolong Wall-clock Time = %e\n", hypre_prolong_wtime);
               hypre_printf("\n");
            }
         }
         else if (dmem_all_data.input.solver == BOOMERAMG ||
                  dmem_all_data.input.solver == BOOMERAMG_MULTADD){
            if (my_id == 0){
               printf("BoomerAMG\n");
            }
            HYPRE_BoomerAMGSetMaxIter(dmem_all_data.hypre.solver, dmem_all_data.input.num_cycles);
            HYPRE_BoomerAMGSetTol(dmem_all_data.hypre.solver, dmem_all_data.input.tol);
            start = MPI_Wtime();
            HYPRE_BoomerAMGSolve(dmem_all_data.hypre.solver,
                                 dmem_all_data.matrix.A_fine,
                                 dmem_all_data.vector_fine.b,
                                 dmem_all_data.vector_fine.x);
            HYPRE_Real amg_wtime = MPI_Wtime() - start;
            HYPRE_BoomerAMGGetNumIterations(dmem_all_data.hypre.solver, &num_iterations);
            HYPRE_BoomerAMGGetFinalRelativeResidualNorm(dmem_all_data.hypre.solver, &final_res_norm);
            if (my_id == 0){
               hypre_printf("\n");
               hypre_printf("Iterations = %d\n", num_iterations);
               hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
               hypre_printf("AMG Wall-clock Time = %e\n", amg_wtime);
               hypre_printf("Comm Wall-clock Time = %e\n", hypre_comm_wtime);
               hypre_printf("Smooth Wall-clock Time = %e\n", hypre_smooth_wtime);
               hypre_printf("Restrict Wall-clock Time = %e\n", hypre_restrict_wtime);
               hypre_printf("Prolong Wall-clock Time = %e\n", hypre_prolong_wtime);
               hypre_printf("\n");
            }
         }
         else {
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
            else if (dmem_all_data.input.solver == SYNC_MULTADD || 
                     dmem_all_data.input.solver == SYNC_AFACX ||
                     dmem_all_data.input.solver == SYNC_AFACJ){
               if (dmem_all_data.input.solver == SYNC_AFACX){
                  if (dmem_all_data.input.oneline_output_flag == 0 && my_id == 0){
                     printf("synchronous AFACx\n\n\n");
                  }
               }
               else if (dmem_all_data.input.solver == SYNC_AFACJ){
                  if (dmem_all_data.input.oneline_output_flag == 0 && my_id == 0){
                     printf("synchronous AFACj\n\n\n");
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
                     if (dmem_all_data.input.multadd_smooth_interp_level_type == SMOOTH_INTERP_AFACJ){
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

   MPI_Finalize();
   return 0;
}
