#ifndef DMEM_MAIN_HPP
#define DMEM_MAIN_HPP

#include "Main.hpp"

#define FINE_INSIDE_SEND 1
#define FINE_INSIDE_RECV 2

#define FINE_OUTSIDE_SEND 3
#define FINE_OUTSIDE_RECV 4

#define GRIDK_INSIDE_SEND 5
#define GRIDK_INSIDE_RECV 6

#define GRIDK_OUTSIDE_SEND 7
#define GRIDK_OUTSIDE_RECV 8

using namespace std;
using namespace mfem;

typedef struct{
   int amr_refs;
   int ref_levels;
   int order;
   char mesh_file[1000];
   double *u;
}DMEM_MfemData;

typedef struct{
   HYPRE_Solver solver_local;
   HYPRE_Solver solver;
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
   hypre_ParVector *x;
   hypre_ParVector *b;
   hypre_ParVector *r;
}DMEM_VectorData;

typedef struct{
   int nx;
   int ny;
   int nz;
   HYPRE_ParCSRMatrix A;
   HYPRE_ParCSRMatrix A_local;
}DMEM_MatrixData;

typedef struct{
   vector<int> procs;
   vector<int> start;
   vector<int> end;
   vector<int> len;
   HYPRE_Real **data;
   MPI_Request *requests;
   int *new_info_flags;
   int type;
   int vector_type;
   int *hypre_map;
}DMEM_CommData;

typedef struct{
   DMEM_CommData gridk_e_inside_send;
   DMEM_CommData gridk_e_inside_recv;

   DMEM_CommData gridk_e_outside_send;
   DMEM_CommData gridk_e_outside_recv;

   DMEM_CommData gridk_r_inside_send;
   DMEM_CommData gridk_r_inside_recv;

   DMEM_CommData gridk_r_outside_send;
   DMEM_CommData gridk_r_outside_recv;

   DMEM_CommData fine_inside_send;
   DMEM_CommData fine_inside_recv;

   DMEM_CommData fine_outside_send;
   DMEM_CommData fine_outside_recv;
   
   HYPRE_Real *fine_send_data;
   HYPRE_Real *fine_recv_data;
   int *hypre_send_map;
   int *hypre_recv_map;
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
   int my_comm;
   int *my_grid_procs_flags;
}DMEM_GridData;

typedef struct{
   DMEM_AllCommData comm;
   DMEM_HypreData hypre;
   DMEM_MatrixData matrix;
   DMEM_VectorData vector;
   DMEM_VectorData *level_vector;
   InputData input;
   OutputData output;
   DMEM_GridData grid;
   DMEM_MfemData mfem;
}DMEM_AllData;

#endif
