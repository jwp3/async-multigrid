#ifndef DMEM_MAIN_HPP
#define DMEM_MAIN_HPP

#include "Main.hpp"

#define FINE_INTRA_INSIDE_SEND 1
#define FINE_INTRA_OUTSIDE_SEND 2

#define FINE_INTRA_INSIDE_RECV 3
#define FINE_INTRA_OUTSIDE_RECV 4

#define GRIDK_INSIDE_SEND 5
#define GRIDK_OUTSIDE_SEND 6

#define GRIDK_INSIDE_RECV 7
#define GRIDK_OUTSIDE_RECV 8

#define READ 1
#define WRITE 2
#define ACCUMULATE 3

#define FINE_INTRA_TAG 1
#define FINEST_TO_GRIDK_CORRECT_TAG 2
#define FINEST_TO_GRIDK_RESIDUAL_TAG 3
#define GRIDJ_TO_GRIDK_CORRECT_TAG 3

#define LOCAL_RES 1
#define GLOBAL_RES 2

#define LOCAL_CONVERGE 1
#define GLOBAL_CONVERGE 2

#define ASSIGN_PROCS_BALANCED_WORK 0
#define ASSIGN_PROCS_SCALAR 1

using namespace std;
using namespace mfem;

typedef struct{
   vector<double> level_wtime;
   double setup_wtime;
   double build_matrix_wtime;
   double solve_wtime;
   double smooth_wtime;
   double residual_wtime;
   double residual_norm_wtime;
   double restrict_wtime;
   double prolong_wtime;
   double correct_wtime;
   double comm_wtime;
   double comp_wtime;
   double coarsest_solve_wtime;
   double start_wtime;
   double end_wtime;
   double inner_solve_wtime;
   double mpiisend_wtime;
   double mpiirecv_wtime;
   double mpitest_wtime;
   double mpiwait_wtime;
   double matvec_wtime;
   double vecop_wtime;
   double r_norm2;
   double r0_norm2;
   double e0_Anorm;
   double e_Anorm;
   double hypre_e_norm2;
   double mfem_e_norm2;
   int num_messages;
}DMEM_OutputData;

typedef struct{
   int num_pre_smooth_sweeps;
   int num_post_smooth_sweeps;
   int num_fine_smooth_sweeps;
   int num_coarse_smooth_sweeps;
   int num_cycles;
   int num_inner_cycles;
   int start_cycle;
   int increment_cycle;
   int num_threads;
   double tol;
   double inner_tol;
   double outer_tol;
   int outer_max_iter;
   int oneline_output_flag;
   int print_reshist_flag;
   int print_output_flag;
   int global_conv_flag;
   double smooth_weight;
   int smoother;
   int solver;
   int outer_solver;
   int async_flag;
   int async_smoother_flag;
   int assign_procs_type;
   double assign_procs_scalar;
   int converge_test_type;
   int res_compute_type;
   int test_problem;
   int hypre_test_error_flag;
   int mfem_test_error_flag;
   int mfem_solve_print_flag;
   int print_level_stats_flag;
   int smooth_interp_type;
   int read_type;
   int print_output_level;
   int coarsest_mult_level;
   int check_res_flag;
   int multadd_smooth_interp_level_type;
   int max_inflight;
   int rhs_type;
   int init_guess_type;
   int afacj_level;
   int num_interpolants;
   int res_update_type;
   int simple_jacobi_flag;
   int sps_probability_type;
   double sps_alpha;
   double sps_min_prob;
   int hypre_memory;
   int async_comm_save_divisor;
   int optimal_jacobi_weight_flag;
   char mat_file_str[1024];
   int eig_CG_max_iters;
   int only_setup_flag;
   int only_build_matrix_flag;
   int include_disconnected_points_flag;
   int eig_power_max_iters;
   int eig_power_MG_max_iters;
   int accel_type;
   int print_matrix_flag;
   int par_file_flag;
   int async_type;
   double imbal;
   double eig_shift;
   double a_eig_shift;
   double b_eig_shift;
   unsigned int delay_usec;
   int delay_id;
   int delay_flag;
   int cheby_grid;
}DMEM_InputData;

typedef struct{
   int amr_refs;
   int ref_levels;
   int par_ref_levels;
   int order;
   int max_amr_iters;
   int max_amr_dofs;
   char mesh_file[1000];
   double *u;
}DMEM_MfemData;

typedef struct{
   HYPRE_Solver solver;
   HYPRE_Solver solver_gridk;
   HYPRE_Solver solver_afacj;
   HYPRE_Solver solver_afacj_gridk;
   HYPRE_Solver outer_solver;
   HYPRE_Int print_level;
   HYPRE_Int interp_type;
   HYPRE_Int coarsen_type;
   HYPRE_Int max_levels;
   HYPRE_Int agg_num_levels;
   HYPRE_Int solver_id;
   HYPRE_Int solve_flag;
   HYPRE_Real strong_threshold;
   HYPRE_Real multadd_trunc_factor;
   HYPRE_Int start_smooth_level;
   HYPRE_Int num_functions;
   HYPRE_Int P_max_elmts;
   HYPRE_Int add_P_max_elmts;
   HYPRE_Real add_trunc_factor;
}DMEM_HypreData;

typedef struct{
   hypre_ParVector *u;
   hypre_ParVector *f;
   hypre_ParVector *x;
   hypre_ParVector *y;
   hypre_ParVector *z;
   hypre_ParVector *b;
   hypre_ParVector *r;
   hypre_ParVector *e;
   hypre_ParVector *d;
   hypre_Vector *x_ghost;
   hypre_Vector *x_ghost_prev;
   hypre_Vector *b_ghost;
   hypre_Vector *a_diag;
   hypre_Vector *a_diag_ghost;
   hypre_Vector *u_ghost;
}DMEM_VectorData;

typedef struct{
   int nx;
   int ny;
   int nz;
   HYPRE_Real difconv_ax;
   HYPRE_Real difconv_ay;
   HYPRE_Real difconv_az;
   HYPRE_Real difconv_cx;
   HYPRE_Real difconv_cy;
   HYPRE_Real difconv_cz;
   int difconv_atype;
   HYPRE_Real vardifconv_eps;
   HYPRE_ParCSRMatrix A_fine;
   HYPRE_ParCSRMatrix A_gridk;
   double **L1_row_norm_gridk;
   double **L1_row_norm_fine;
   double **wJacobi_scale_gridk;
   double **wJacobi_scale_fine;
   double **symmL1_row_norm_gridk;
   double **symmL1_row_norm_fine;
   double **symmwJacobi_scale_gridk;
   double **symmwJacobi_scale_fine;
   hypre_ParCSRMatrix *P_gridk;
   hypre_ParCSRMatrix *R_gridk;
   hypre_ParCSRMatrix **P_fine;
   hypre_ParCSRMatrix **R_fine;
}DMEM_MatrixData;

typedef struct{
   vector<int> procs;
   vector<int> message_count;
   vector<int> done_flags;
   vector<int> recv_flags;
   vector<int> start;
   vector<int> end;
   vector<int> len;
   vector<int> next_inflight;
   vector<int> num_inflight;
   vector<int> max_inflight;
   vector<int> semi_async_flags;
   vector<HYPRE_Real> r_norm;
   vector<HYPRE_Real> r_norm_boundary;
   vector<HYPRE_Real> r_norm_boundary_prev;
   HYPRE_Real **data;
   HYPRE_Real ***data_inflight;
   MPI_Request *requests;
   MPI_Request **requests_inflight;
   int **inflight_flags;
   int type;
   int tag;
   vector<vector<vector<HYPRE_Real>>> a_ghost_data;
   vector<vector<vector<HYPRE_Int>>> a_ghost_j;
   int update_res_in_comm;
}DMEM_CommData;

typedef struct{
  // DMEM_CommData gridk_e_inside_send;
  // DMEM_CommData gridk_e_inside_recv;

  // DMEM_CommData gridk_e_outside_send;
  // DMEM_CommData gridk_e_outside_recv;

  // DMEM_CommData gridk_r_inside_send;
  // DMEM_CommData gridk_r_inside_recv;

  // DMEM_CommData gridk_r_outside_send;
  // DMEM_CommData gridk_r_outside_recv;

  // DMEM_CommData fine_inside_send;
  // DMEM_CommData fine_inside_recv;

  // DMEM_CommData fine_outside_send;
  // DMEM_CommData fine_outside_recv;
   
   DMEM_CommData finestToGridk_Correct_insideSend;
   DMEM_CommData finestToGridk_Correct_insideRecv;

   DMEM_CommData finestToGridk_Correct_outsideSend;
   DMEM_CommData finestToGridk_Correct_outsideRecv;

   DMEM_CommData finestToGridk_Residual_insideSend;
   DMEM_CommData finestToGridk_Residual_insideRecv;

   DMEM_CommData finestToGridk_Residual_outsideSend;
   DMEM_CommData finestToGridk_Residual_outsideRecv;

   DMEM_CommData finestIntra_insideSend;
   DMEM_CommData finestIntra_insideRecv;

   DMEM_CommData finestIntra_outsideSend;
   DMEM_CommData finestIntra_outsideRecv;



   DMEM_CommData gridjToGridk_Correct_insideSend;
   DMEM_CommData gridjToGridk_Correct_insideRecv;
 
   DMEM_CommData gridjToGridk_Correct_outsideSend;
   DMEM_CommData gridjToGridk_Correct_outsideRecv;

   HYPRE_Real *fine_send_data;
   HYPRE_Real *fine_recv_data;
   int *hypre_send_map;
   int *hypre_recv_map;
   int all_done_flag;
   int outside_done_flag;
   int outside_recv_done_flag;
   int async_smooth_done_flag;
   int is_async_smoothing_flag;
}DMEM_AllCommData;

typedef struct{
   int num_levels;
   int *n;
   double tot_work;
   double *level_work;
   int *num_procs_level;
   double *frac_level_work;
   int **procs;
   int my_grid;
   MPI_Comm my_comm;
   int *my_grid_procs_flags;
}DMEM_GridData;

typedef struct{
   double r_L2norm_local;
   double r_L1norm_local;
   int r_L2norm_local_converge_flag;
   int cycle;
   int inner_cycle;
   int relax;
   int converge_flag;
   int grid_done_flag;
}DMEM_IterData;

typedef struct{
   double alpha;
   double beta;
   double mu;
   double delta;
   double c_prev;
   double c;
   double c_next;
}DMEM_ChebyData;

typedef struct{
   DMEM_AllCommData comm;
   DMEM_HypreData hypre;
   DMEM_MatrixData matrix;
   DMEM_VectorData vector_fine;
   DMEM_VectorData vector_gridk;
   DMEM_InputData input;
   DMEM_OutputData output;
   DMEM_GridData grid;
   DMEM_MfemData mfem;
   DMEM_IterData iter;
   DMEM_ChebyData cheby;
}DMEM_AllData;

extern HYPRE_Int vecop_machine;

#endif
