#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Setup.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_Add.hpp"
#include "DMEM_Mult.hpp"

int main (int argc, char *argv[])
{
   int myid, num_procs;
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
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Hypre parameters */
   dmem_all_data.hypre.max_levels = 20;
   dmem_all_data.hypre.solver_id = 0;
   dmem_all_data.hypre.agg_num_levels = 0;
   dmem_all_data.hypre.coarsen_type = 8; /* for proc-independent coarsening, can also use 0, 7, or 9 with "-agg_nl 0" */
   dmem_all_data.hypre.interp_type = 6;
   dmem_all_data.hypre.print_level = 0;
   dmem_all_data.hypre.solve_flag = 0;
   dmem_all_data.hypre.strong_threshold = .25;

   int n = 10;
   dmem_all_data.matrix.nx = n;
   dmem_all_data.matrix.ny = n;
   dmem_all_data.matrix.nz = n;
   /* mfem parameters */
   dmem_all_data.mfem.ref_levels = 4;
   dmem_all_data.mfem.order = 1;
   strcpy(dmem_all_data.mfem.mesh_file, "./mfem/data/ball-nurbs.mesh");
   dmem_all_data.mfem.amr_refs = 0;

   dmem_all_data.input.test_problem = LAPLACE_3D27PT;
   dmem_all_data.input.tol = 1e-9;
   dmem_all_data.input.async_flag = 0;
   dmem_all_data.input.async_type = FULL_ASYNC;
   dmem_all_data.input.global_conv_flag = 0;
   dmem_all_data.input.thread_part_type = ALL_LEVELS;
   dmem_all_data.input.converge_test_type = LOCAL;
   dmem_all_data.input.res_compute_type = LOCAL;
   dmem_all_data.input.num_pre_smooth_sweeps = 1;
   dmem_all_data.input.num_post_smooth_sweeps = 1;
   dmem_all_data.input.num_fine_smooth_sweeps = 1;
   dmem_all_data.input.num_coarse_smooth_sweeps = 1;
   dmem_all_data.input.format_output_flag = 0;
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
   dmem_all_data.input.num_cycles = 20;
   dmem_all_data.input.increment_cycle = 1;
   dmem_all_data.input.start_cycle = 1;

   /* Parse command line */
   int arg_index = 0;
   int print_usage = 0;

   while (arg_index < argc)
   {
      if (strcmp(argv[arg_index], "-n") == 0)
      {
         arg_index++;
	 n = atoi(argv[arg_index]);
	 dmem_all_data.matrix.nx = n;
         dmem_all_data.matrix.ny = n;
         dmem_all_data.matrix.nz = n;
      }
      else if (strcmp(argv[arg_index], "-nx") == 0)
      {
         arg_index++;
         dmem_all_data.matrix.nx = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-ny") == 0)
      {
         arg_index++;
         dmem_all_data.matrix.ny = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-nz") == 0)
      {
         arg_index++;
         dmem_all_data.matrix.nz = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-problem") == 0)
      {
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
         else if (strcmp(argv[arg_index], "mfem_laplace") == 0){
            dmem_all_data.input.test_problem = MFEM_LAPLACE;
         }
         else if (strcmp(argv[arg_index], "mfem_elast") == 0){
            dmem_all_data.input.test_problem = MFEM_ELAST;
         }
	 else if (strcmp(argv[arg_index], "mfem_maxwell") == 0){
            dmem_all_data.input.test_problem = MFEM_MAXWELL;
         }
      }
      else if (strcmp(argv[arg_index], "-smoother") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "j") == 0){
            dmem_all_data.input.smoother = JACOBI;
            dmem_all_data.input.smooth_interp_type = JACOBI;
         }
         else if (strcmp(argv[arg_index], "gs") == 0){
            dmem_all_data.input.smoother = GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "hybrid_jgs") == 0){
            dmem_all_data.input.smoother = HYBRID_JACOBI_GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "async_gs") == 0){
            dmem_all_data.input.smoother = ASYNC_GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "semi_async_gs") == 0){
            dmem_all_data.input.smoother = SEMI_ASYNC_GAUSS_SEIDEL;
         }
         else if (strcmp(argv[arg_index], "L1j") == 0){
            dmem_all_data.input.smoother = L1_JACOBI;
            dmem_all_data.input.smooth_interp_type = L1_JACOBI;
         }
      }
      else if (strcmp(argv[arg_index], "-solver") == 0)
      {
         arg_index++;
         dmem_all_data.input.solver = atoi(argv[arg_index]);
         if (strcmp(argv[arg_index], "mult") == 0){
            dmem_all_data.input.solver = MULT;
         }
         else if (strcmp(argv[arg_index], "multadd") == 0){
            dmem_all_data.input.solver = MULTADD;
         }
         else if (strcmp(argv[arg_index], "afacx") == 0){
            dmem_all_data.input.solver = AFACX;
         }
         else if (strcmp(argv[arg_index], "async_multadd") == 0){
            dmem_all_data.input.solver = ASYNC_MULTADD;
            dmem_all_data.input.async_flag = 1;
         }
         else if (strcmp(argv[arg_index], "async_afacx") == 0){
            dmem_all_data.input.solver = ASYNC_AFACX;
            dmem_all_data.input.async_flag = 1;
         }
      }
      else if (strcmp(argv[arg_index], "-smooth_interp") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "j") == 0){
            dmem_all_data.input.smooth_interp_type = JACOBI;
         }
         else if (strcmp(argv[arg_index], "L1j") == 0){
            dmem_all_data.input.smooth_interp_type = L1_JACOBI;
         }
      }
      else if (strcmp(argv[arg_index], "-smooth_weight") == 0)
      {
         arg_index++;
         dmem_all_data.input.smooth_weight = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-th") == 0)
      {
         arg_index++;
         dmem_all_data.hypre.strong_threshold = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_cycles") == 0)
      {
         arg_index++;
         dmem_all_data.input.num_cycles = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-start_cycle") == 0)
      {
         arg_index++;
         dmem_all_data.input.start_cycle = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-incr_cycle") == 0)
      {
         arg_index++;
         dmem_all_data.input.increment_cycle = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-tol") == 0)
      {
         arg_index++;
         dmem_all_data.input.tol = atof(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_smooth_sweeps") == 0)
      {
         arg_index++;
         dmem_all_data.input.num_pre_smooth_sweeps = atoi(argv[arg_index]);
         dmem_all_data.input.num_post_smooth_sweeps = atoi(argv[arg_index]);
         dmem_all_data.input.num_fine_smooth_sweeps = atoi(argv[arg_index]);
         dmem_all_data.input.num_coarse_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_pre_smooth_sweeps") == 0)
      {
         arg_index++;
         dmem_all_data.input.num_pre_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-num_post_smooth_sweeps") == 0)
      {
         arg_index++;
         dmem_all_data.input.num_post_smooth_sweeps = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mxl") == 0)
      {
         arg_index++;
         dmem_all_data.hypre.max_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-agg_nl") == 0)
      {
         arg_index++;
         dmem_all_data.hypre.agg_num_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-coarsen_type") == 0)
      {
         arg_index++;
         dmem_all_data.hypre.coarsen_type = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-interp_type") == 0)
      {
         arg_index++;
         dmem_all_data.hypre.interp_type = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_ref_levels") == 0)
      {
         arg_index++;
         dmem_all_data.mfem.ref_levels = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_amr_refs") == 0)
      {
         arg_index++;
         dmem_all_data.mfem.amr_refs = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_order") == 0)
      {
         arg_index++;
         dmem_all_data.mfem.order = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-mfem_mesh_file") == 0)
      {
         arg_index++;
         strcpy(dmem_all_data.mfem.mesh_file, argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-converge_test_type") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "local") == 0){
            dmem_all_data.input.converge_test_type = LOCAL;
            dmem_all_data.input.global_conv_flag = 0;
         }
         else if (strcmp(argv[arg_index], "global") == 0){
            dmem_all_data.input.converge_test_type = GLOBAL;
            dmem_all_data.input.global_conv_flag = 1;
         }
      }
      else if (strcmp(argv[arg_index], "-num_runs") == 0)
      {
         arg_index++;
         num_runs = atoi(argv[arg_index]);
      }
      else if (strcmp(argv[arg_index], "-read_type") == 0)
      {
         arg_index++;
         if (strcmp(argv[arg_index], "sol") == 0){
            dmem_all_data.input.read_type = READ_SOL;
         }
         else if (strcmp(argv[arg_index], "res") == 0){
            dmem_all_data.input.read_type = READ_RES;
         }
      }
      else if (strcmp(argv[arg_index], "-no_output") == 0)
      {
         dmem_all_data.input.print_output_flag = 0;
      }
      else if (strcmp(argv[arg_index], "-format_output") == 0)
      {
         dmem_all_data.input.format_output_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_reshist") == 0)
      {
         dmem_all_data.input.print_reshist_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-print_hypre") == 0)
      {
         dmem_all_data.hypre.print_level = 3;
      }
      else if (strcmp(argv[arg_index], "-print_level_stats") == 0)
      {
         dmem_all_data.input.print_level_stats_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-hypre_test_error") == 0)
      {
         dmem_all_data.input.hypre_test_error_flag = 1;
      }
      else if (strcmp(argv[arg_index], "-mfem_test_error") == 0)
      {
         dmem_all_data.input.mfem_test_error_flag = 1;
      }
      arg_index++;
   }
   
   srand(0);
  // mkl_set_num_threads(1);
 
   start = omp_get_wtime(); 
   DMEM_Setup(&dmem_all_data);
  // DMEM_ResetData(&dmem_all_data);
   dmem_all_data.output.setup_wtime = omp_get_wtime() - start;

  // DMEM_Async_AMG(&dmem_all_data);
   DMEM_Mult(&dmem_all_data);   

   DMEM_PrintOutput(&dmem_all_data);

   MPI_Finalize();
   return 0;
}
