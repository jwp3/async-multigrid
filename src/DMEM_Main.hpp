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
#define ACCUMULATE 2

#define FINE_INTRA_TAG 1
#define FINEST_TO_GRIDK_CORRECT_TAG 2
#define FINEST_TO_GRIDK_RESIDUAL_TAG 3
#define GRIDJ_TO_GRIDK_CORRECT_TAG 3

#define LOCAL_RES 1
#define GLOBAL_RES 2

#define LOCAL_CONVERGE 1
#define GLOBAL_CONVERGE 2

using namespace std;
using namespace mfem;

typedef struct{
   int *smooth_relax;
   int *smooth_sweeps;
   int *cycles;
   double setup_wtime;
   double solve_wtime;
   double smooth_wtime;
   double residual_wtime;
   double residual_norm_wtime;
   double restrict_wtime;
   double prolong_wtime;
   double correct_wtime;
   double comm_wtime;
   double coarsest_solve_wtime;
   double start_wtime;
   double end_wtime;
   double r_norm2;
   double r0_norm2;
   double hypre_e_norm2;
   double mfem_e_norm2;
}DMEM_OutputData;

typedef struct{
   int num_pre_smooth_sweeps;
   int num_post_smooth_sweeps;
   int num_fine_smooth_sweeps;
   int num_coarse_smooth_sweeps;
   int num_cycles;
   int start_cycle;
   int increment_cycle;
   int num_threads;
   double tol;
   int oneline_output_flag;
   int print_reshist_flag;
   int print_output_flag;
   int global_conv_flag;
   double smooth_weight;
   int smoother;
   int solver;
   int async_flag;
   int async_type;
   int thread_part_type;
   int thread_part_distr_type;
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
}DMEM_InputData;


typedef struct{
   int amr_refs;
   int ref_levels;
   int par_ref_levels;
   int order;
   char mesh_file[1000];
   double *u;
}DMEM_MfemData;

typedef struct{
   HYPRE_Solver solver;
   HYPRE_Solver solver_gridk;
   HYPRE_Int print_level;
   HYPRE_Int interp_type;
   HYPRE_Int coarsen_type;
   HYPRE_Int max_levels;
   HYPRE_Int agg_num_levels;
   HYPRE_Int solver_id;
   HYPRE_Int solve_flag;
   HYPRE_Real strong_threshold;
}DMEM_HypreData;

typedef struct{
   hypre_ParVector *u;
   hypre_ParVector *f;
   hypre_ParVector *x;
   hypre_ParVector *b;
   hypre_ParVector *r;
   hypre_ParVector *e;
   hypre_Vector *x_ghost;
}DMEM_VectorData;

typedef struct{
   int nx;
   int ny;
   int nz;
   HYPRE_ParCSRMatrix A_fine;
   HYPRE_ParCSRMatrix A_gridk;
}DMEM_MatrixData;

typedef struct{
   vector<int> procs;
   vector<int> message_count;
   vector<int> done_flags;
   vector<int> start;
   vector<int> end;
   vector<int> len;
   HYPRE_Real **data;
   MPI_Request *requests;
   int *new_info_flags;
   int type;
   int vector_type;
   int tag;
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
}DMEM_AllCommData;

typedef struct{
   int num_levels;
   int *n;
   int tot_work;
   int *level_work;
   int *num_procs_level;
   double *frac_level_work;
   int **procs;
   int my_grid;
   MPI_Comm my_comm;
   int *my_grid_procs_flags;
}DMEM_GridData;

typedef struct{
   double r_norm2_local;
   int r_norm2_local_converge_flag;
   int cycle;
}DMEM_IterData;

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
}DMEM_AllData;

#endif
