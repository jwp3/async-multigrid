#include "Main.hpp"
#include "DMEM_Setup.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_Laplacian.hpp"
#include "Misc.hpp"

using namespace std;

void SetHypreSolver(DMEM_AllData *dmem_all_data,
                    HYPRE_Solver *solver);
void ConstructVectors(DMEM_AllData *dmem_all_data,
                 hypre_ParCSRMatrix *A,
                 DMEM_VectorData *vector);
void ComputeWork(DMEM_AllData *all_data);
void PartitionProcs(DMEM_AllData *dmem_all_data);
void PartitionGrids(DMEM_AllData *dmem_all_data);
void ConstructMatrix(DMEM_AllData *dmem_all_data,
                     HYPRE_ParCSRMatrix *A,
                     MPI_Comm comm);
void CreateCommData(DMEM_AllData *dmem_all_data);

//TODO: clear extra memory
void DMEM_Setup(DMEM_AllData *dmem_all_data)
{
   double start;
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    
   start = omp_get_wtime();
   /* fine */
   ConstructMatrix(dmem_all_data,
		   &(dmem_all_data->matrix.A_fine),
		   MPI_COMM_WORLD);
   ConstructVectors(dmem_all_data,
                    dmem_all_data->matrix.A_fine,
                    &(dmem_all_data->vector_fine));
   SetHypreSolver(dmem_all_data,
                  &(dmem_all_data->hypre.solver));
   HYPRE_BoomerAMGSetup(dmem_all_data->hypre.solver,
			dmem_all_data->matrix.A_fine,
			dmem_all_data->vector_fine.b,
			dmem_all_data->vector_fine.x);

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   dmem_all_data->grid.num_levels = hypre_ParAMGDataNumLevels(amg_data);
  // if (my_id == 0)
  // printf("num levels %d\n", hypre_ParAMGDataNumLevels(amg_data));
  // MPI_Barrier(MPI_COMM_WORLD);

  // HYPRE_BoomerAMGDestroy(dmem_all_data->hypre.solver);

   /* gridk */
//   PartitionProcs(dmem_all_data);
//   ConstructMatrix(dmem_all_data, 
//		   &(dmem_all_data->matrix.A_gridk),
//		   dmem_all_data->grid.my_comm);
//   ConstructVectors(dmem_all_data,
//                    dmem_all_data->matrix.A_gridk,
//                    &(dmem_all_data->vector_gridk));
//   SetHypreSolver(dmem_all_data,
//                  &(dmem_all_data->hypre.solver_gridk));
//   HYPRE_BoomerAMGSetup(dmem_all_data->hypre.solver_gridk,
//                        dmem_all_data->matrix.A_gridk,
//                        dmem_all_data->vector_gridk.b,
//                        dmem_all_data->vector_gridk.x);
//   amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
//  // if (my_id == 0)
//  // printf("num levels %d\n", hypre_ParAMGDataNumLevels(amg_data));
//  // MPI_Barrier(MPI_COMM_WORLD);
//   dmem_all_data->output.hypre_setup_wtime = omp_get_wtime() - start;
//   CreateCommData(dmem_all_data);

   hypre_ParVector *r = dmem_all_data->vector_fine.r;
   dmem_all_data->output.r0_norm2 =
      sqrt(hypre_ParVectorInnerProd(r, r));
}

void SetHypreSolver(DMEM_AllData *dmem_all_data,
		    HYPRE_Solver *solver)
{
   HYPRE_BoomerAMGCreate(solver);
   HYPRE_BoomerAMGSetPrintLevel(*solver, dmem_all_data->hypre.print_level);
   HYPRE_BoomerAMGSetOldDefault(*solver);
   HYPRE_BoomerAMGSetPostInterpType(*solver, 0);
   HYPRE_BoomerAMGSetInterpType(*solver, dmem_all_data->hypre.interp_type);
   HYPRE_BoomerAMGSetRestriction(*solver, 0);
   HYPRE_BoomerAMGSetCoarsenType(*solver, dmem_all_data->hypre.coarsen_type);
   HYPRE_BoomerAMGSetMaxLevels(*solver, dmem_all_data->hypre.max_levels);
   HYPRE_BoomerAMGSetAggNumLevels(*solver, dmem_all_data->hypre.agg_num_levels);
   HYPRE_BoomerAMGSetRelaxType(*solver, 0);
   HYPRE_BoomerAMGSetRelaxWt(*solver, dmem_all_data->input.smooth_weight);
   HYPRE_BoomerAMGSetCycleRelaxType(*solver, 99, 3);

   /* multadd options */
 //  HYPRE_BoomerAMGSetMultAdditive(*solver, 0);
 //  HYPRE_BoomerAMGSetMultAddTruncFactor(*solver, 0);
 //  HYPRE_BoomerAMGSetMultAddPMaxElmts(*solver, hypre_ParCSRMatrixNumRows(dmem_all_data->matrix.A_fine));
 // // printf("%d\n", hypre_ParAMGDataAddRelaxType(*solver));
 //  HYPRE_BoomerAMGSetAddRelaxType(*solver, 0);
 //  HYPRE_BoomerAMGSetAddRelaxWt(*solver, dmem_all_data->input.smooth_weight);
 //  
 // // HYPRE_BoomerAMGSetAddLastLvl(*solver,
 // //                              dmem_all_data->hypre.max_levels);
}

void AllocCommVars(DMEM_CommData *comm_data)
{
   comm_data->len.resize(comm_data->procs.size());
   comm_data->start.resize(comm_data->procs.size());
   comm_data->end.resize(comm_data->procs.size());
   comm_data->requests = (MPI_Request *)malloc(comm_data->procs.size() * sizeof(MPI_Request));
   comm_data->new_info_flags = (int *)calloc(comm_data->procs.size(), sizeof(int));
}

void CreateCommData(DMEM_AllData *dmem_all_data)
{
   int num_procs, my_id;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);   

   int my_grid = dmem_all_data->grid.my_grid;

   int gridk_start = hypre_ParCSRMatrixFirstRowIndex(dmem_all_data->matrix.A_gridk);
   int gridk_end = hypre_ParCSRMatrixLastRowIndex(dmem_all_data->matrix.A_gridk);
   int my_gridk_part[2] = {gridk_start, gridk_end};
   int all_gridk_parts[2*num_procs];
   MPI_Allgather(my_gridk_part, 2, MPI_INT, all_gridk_parts, 2, MPI_INT, MPI_COMM_WORLD);

   int fine_start = hypre_ParCSRMatrixFirstRowIndex(dmem_all_data->matrix.A_fine);
   int fine_end = hypre_ParCSRMatrixLastRowIndex(dmem_all_data->matrix.A_fine);
   int my_fine_part[2] = {fine_start, fine_end};
   int all_fine_parts[2*num_procs];
   MPI_Allgather(my_fine_part, 2, MPI_INT, all_fine_parts, 2, MPI_INT, MPI_COMM_WORLD);

   dmem_all_data->grid.my_grid_procs_flags = (int *)calloc(num_procs, sizeof(int));
   my_grid = dmem_all_data->grid.my_grid;
   for (int i = 0; i < dmem_all_data->grid.num_procs_level[my_grid]; i++){
      int ip = dmem_all_data->grid.procs[my_grid][i];
      dmem_all_data->grid.my_grid_procs_flags[ip] = 1;
   }

/**************************************
 * gridk res
 **************************************/

/* gridk res inside send */
   dmem_all_data->comm.gridk_r_inside_send.type = GRIDK_INSIDE_SEND;
   for (int p = 0; p < num_procs; p++){
      if (/*p != my_id && */dmem_all_data->grid.my_grid_procs_flags[p] == 1){
         int p_gridk_start = all_gridk_parts[2*p];
         int p_gridk_end = all_gridk_parts[2*p+1];
         int flag = 0;
         if (p_gridk_start >= fine_start && p_gridk_start <= fine_end){
            flag = 1;
         }
         else if (p_gridk_end <= fine_end && p_gridk_end >= fine_start){
            flag = 1;
         }
         else if (p_gridk_end >= fine_end && p_gridk_start <= fine_start){
            flag = 1;
         }
         if (flag == 1){
            dmem_all_data->comm.gridk_r_inside_send.procs.push_back(p);
         }
      }
   }

   dmem_all_data->comm.gridk_r_inside_send.data =
      (HYPRE_Real **)malloc(dmem_all_data->comm.gridk_r_inside_send.procs.size() * sizeof(HYPRE_Real *));
   AllocCommVars(&(dmem_all_data->comm.gridk_r_inside_send));
 
   for (int i = 0; i < dmem_all_data->comm.gridk_r_inside_send.procs.size(); i++){
      int p = dmem_all_data->comm.gridk_r_inside_send.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (p_gridk_start >= fine_start && p_gridk_start <= fine_end){
         dmem_all_data->comm.gridk_r_inside_send.start[i] = p_gridk_start;
         dmem_all_data->comm.gridk_r_inside_send.end[i] = fine_end;
      }
      else if (p_gridk_end <= fine_end && p_gridk_end >= fine_start){
         dmem_all_data->comm.gridk_r_inside_send.start[i] = fine_start;
         dmem_all_data->comm.gridk_r_inside_send.end[i] = p_gridk_end;
      }
      else if (p_gridk_end >= fine_end && p_gridk_start <= fine_start){
         dmem_all_data->comm.gridk_r_inside_send.start[i] = fine_start;
         dmem_all_data->comm.gridk_r_inside_send.end[i] = fine_end;
      }

      dmem_all_data->comm.gridk_r_inside_send.len[i] =
         dmem_all_data->comm.gridk_r_inside_send.end[i] - dmem_all_data->comm.gridk_r_inside_send.start[i] + 1;
      dmem_all_data->comm.gridk_r_inside_send.start[i] -= fine_start;
      dmem_all_data->comm.gridk_r_inside_send.end[i] -= fine_start;
      dmem_all_data->comm.gridk_r_inside_send.data[i] =
         (HYPRE_Real *)calloc(dmem_all_data->comm.gridk_r_inside_send.len[i], sizeof(HYPRE_Real));
   }

/* gridk res outside send */
   dmem_all_data->comm.gridk_r_outside_send.type = GRIDK_OUTSIDE_SEND;
   for (int p = 0; p < num_procs; p++){
      if (/*p != my_id && */dmem_all_data->grid.my_grid_procs_flags[p] == 0){
         int p_gridk_start = all_gridk_parts[2*p];
         int p_gridk_end = all_gridk_parts[2*p+1];
         int flag = 0;
         if (p_gridk_start >= fine_start && p_gridk_start <= fine_end){
            flag = 1;
         }
         else if (p_gridk_end <= fine_end && p_gridk_end >= fine_start){
            flag = 1;
         }
         else if (p_gridk_end >= fine_end && p_gridk_start <= fine_start){
            flag = 1;
         }
         if (flag == 1){
            dmem_all_data->comm.gridk_r_outside_send.procs.push_back(p);
         }
      }
   }

   dmem_all_data->comm.gridk_r_outside_send.data =
      (HYPRE_Real **)malloc(dmem_all_data->comm.gridk_r_outside_send.procs.size() * sizeof(HYPRE_Real *));
   AllocCommVars(&(dmem_all_data->comm.gridk_r_outside_send));

   for (int i = 0; i < dmem_all_data->comm.gridk_r_outside_send.procs.size(); i++){
      int p = dmem_all_data->comm.gridk_r_outside_send.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (p_gridk_start >= fine_start && p_gridk_start <= fine_end){
         dmem_all_data->comm.gridk_r_outside_send.start[i] = p_gridk_start;
         dmem_all_data->comm.gridk_r_outside_send.end[i] = fine_end;
      }
      else if (p_gridk_end <= fine_end && p_gridk_end >= fine_start){
         dmem_all_data->comm.gridk_r_outside_send.start[i] = fine_start;
         dmem_all_data->comm.gridk_r_outside_send.end[i] = p_gridk_end;
      }
      else if (p_gridk_end >= fine_end && p_gridk_start <= fine_start){
         dmem_all_data->comm.gridk_r_outside_send.start[i] = fine_start;
         dmem_all_data->comm.gridk_r_outside_send.end[i] = fine_end;
      }

      dmem_all_data->comm.gridk_r_outside_send.len[i] =
         dmem_all_data->comm.gridk_r_outside_send.end[i] - dmem_all_data->comm.gridk_r_outside_send.start[i] + 1;
      dmem_all_data->comm.gridk_r_outside_send.start[i] -= fine_start;
      dmem_all_data->comm.gridk_r_outside_send.end[i] -= fine_start;
      dmem_all_data->comm.gridk_r_outside_send.data[i] =
         (HYPRE_Real *)calloc(dmem_all_data->comm.gridk_r_outside_send.len[i], sizeof(HYPRE_Real));
   }

/* gridk res inside recv */
   dmem_all_data->comm.gridk_r_inside_recv.type = GRIDK_INSIDE_RECV;
   for (int p = 0; p < num_procs; p++){
      if (/*p != my_id && */dmem_all_data->grid.my_grid_procs_flags[p] == 1){
         int flag = 0;
         int p_fine_start = all_fine_parts[2*p];
         int p_fine_end = all_fine_parts[2*p+1];
         if (gridk_start >= p_fine_start && gridk_start <= p_fine_end){
            flag = 1;
         }
         else if (gridk_end <= p_fine_end && gridk_end >= p_fine_start){
            flag = 1;
         }
         else if (gridk_start <= p_fine_start && gridk_end >= p_fine_end){
            flag = 1;
         }
         if (flag == 1){
            dmem_all_data->comm.gridk_r_inside_recv.procs.push_back(p);
         }
      }
   }

   dmem_all_data->comm.gridk_r_inside_recv.data =
      (HYPRE_Real **)malloc(dmem_all_data->comm.gridk_r_inside_recv.procs.size() * sizeof(HYPRE_Real *));
   AllocCommVars(&(dmem_all_data->comm.gridk_r_inside_recv));

   for (int i = 0; i < dmem_all_data->comm.gridk_r_inside_recv.procs.size(); i++){
      int p = dmem_all_data->comm.gridk_r_inside_recv.procs[i];
      int p_fine_start = all_fine_parts[2*p];
      int p_fine_end = all_fine_parts[2*p+1];
      if (gridk_start >= p_fine_start && gridk_start <= p_fine_end){
         dmem_all_data->comm.gridk_r_inside_recv.start[i] = gridk_start;
         dmem_all_data->comm.gridk_r_inside_recv.end[i] = p_fine_end;
      }
      else if (gridk_end <= p_fine_end && gridk_end >= p_fine_start){
         dmem_all_data->comm.gridk_r_inside_recv.start[i] = p_fine_start;
         dmem_all_data->comm.gridk_r_inside_recv.end[i] = gridk_end;
      }
      else if (gridk_start <= p_fine_start && gridk_end >= p_fine_end){
         dmem_all_data->comm.gridk_r_inside_recv.start[i] = p_fine_start;
         dmem_all_data->comm.gridk_r_inside_recv.end[i] = p_fine_end;
      }

      dmem_all_data->comm.gridk_r_inside_recv.len[i] =
         dmem_all_data->comm.gridk_r_inside_recv.end[i] - dmem_all_data->comm.gridk_r_inside_recv.start[i] + 1;
      dmem_all_data->comm.gridk_r_inside_recv.start[i] -= gridk_start;
      dmem_all_data->comm.gridk_r_inside_recv.end[i] -= gridk_start;
      dmem_all_data->comm.gridk_r_inside_recv.data[i] =
         (HYPRE_Real *)calloc(dmem_all_data->comm.gridk_r_inside_recv.len[i], sizeof(HYPRE_Real));
   }


/* gridk res outside recv */
   dmem_all_data->comm.gridk_r_outside_recv.type = GRIDK_OUTSIDE_RECV;
   for (int p = 0; p < num_procs; p++){
      if (/*p != my_id && */dmem_all_data->grid.my_grid_procs_flags[p] == 0){
         int flag = 0;
         int p_fine_start = all_fine_parts[2*p];
         int p_fine_end = all_fine_parts[2*p+1];
         if (gridk_start >= p_fine_start && gridk_start <= p_fine_end){
            flag = 1;
         }
         else if (gridk_end <= p_fine_end && gridk_end >= p_fine_start){
            flag = 1;
         }
         else if (gridk_start <= p_fine_start && gridk_end >= p_fine_end){
            flag = 1;
         }
         if (flag == 1){
            dmem_all_data->comm.gridk_r_outside_recv.procs.push_back(p);
         }
      }
   }

   dmem_all_data->comm.gridk_r_outside_recv.data =
      (HYPRE_Real **)malloc(dmem_all_data->comm.gridk_r_outside_recv.procs.size() * sizeof(HYPRE_Real *));
   AllocCommVars(&(dmem_all_data->comm.gridk_r_outside_recv));

   for (int i = 0; i < dmem_all_data->comm.gridk_r_outside_recv.procs.size(); i++){
      int p = dmem_all_data->comm.gridk_r_outside_recv.procs[i];
      int p_fine_start = all_fine_parts[2*p];
      int p_fine_end = all_fine_parts[2*p+1];
      if (gridk_start >= p_fine_start && gridk_start <= p_fine_end){
         dmem_all_data->comm.gridk_r_outside_recv.start[i] = gridk_start;
         dmem_all_data->comm.gridk_r_outside_recv.end[i] = p_fine_end;
      }
      else if (gridk_end <= p_fine_end && gridk_end >= p_fine_start){
         dmem_all_data->comm.gridk_r_outside_recv.start[i] = p_fine_start;
         dmem_all_data->comm.gridk_r_outside_recv.end[i] = gridk_end;
      }
      else if (gridk_start <= p_fine_start && gridk_end >= p_fine_end){
         dmem_all_data->comm.gridk_r_outside_recv.start[i] = p_fine_start;
         dmem_all_data->comm.gridk_r_outside_recv.end[i] = p_fine_end;
      }

      dmem_all_data->comm.gridk_r_outside_recv.len[i] =
         dmem_all_data->comm.gridk_r_outside_recv.end[i] - dmem_all_data->comm.gridk_r_outside_recv.start[i] + 1;
      dmem_all_data->comm.gridk_r_outside_recv.start[i] -= gridk_start;
      dmem_all_data->comm.gridk_r_outside_recv.end[i] -= gridk_start;
      dmem_all_data->comm.gridk_r_outside_recv.data[i] =
         (HYPRE_Real *)calloc(dmem_all_data->comm.gridk_r_outside_recv.len[i], sizeof(HYPRE_Real));
   }

/**************************************
 * gridk correct
 **************************************/

/* gridk correct inside send */
   dmem_all_data->comm.gridk_e_inside_send.type = GRIDK_INSIDE_SEND;

   dmem_all_data->comm.gridk_e_inside_send.procs = dmem_all_data->comm.gridk_r_inside_recv.procs;
   dmem_all_data->comm.gridk_e_inside_send.data =
      (HYPRE_Real **)malloc(dmem_all_data->comm.gridk_e_inside_send.procs.size() * sizeof(HYPRE_Real *));
   AllocCommVars(&(dmem_all_data->comm.gridk_e_inside_send));
 
   for (int i = 0; i < dmem_all_data->comm.gridk_e_inside_send.procs.size(); i++){
      dmem_all_data->comm.gridk_e_inside_send.start[i] = dmem_all_data->comm.gridk_r_inside_recv.start[i];
      dmem_all_data->comm.gridk_e_inside_send.end[i] = dmem_all_data->comm.gridk_r_inside_recv.end[i];
      dmem_all_data->comm.gridk_e_inside_send.len[i] = dmem_all_data->comm.gridk_r_inside_recv.len[i];
      dmem_all_data->comm.gridk_e_inside_send.data[i] =
         (HYPRE_Real *)calloc(dmem_all_data->comm.gridk_e_inside_send.len[i], sizeof(HYPRE_Real));
   }

/* gridk correct outside send */
   dmem_all_data->comm.gridk_e_outside_send.type = GRIDK_OUTSIDE_SEND;
   
   dmem_all_data->comm.gridk_e_outside_send.procs = dmem_all_data->comm.gridk_r_outside_recv.procs;
   dmem_all_data->comm.gridk_e_outside_send.data =
      (HYPRE_Real **)malloc(dmem_all_data->comm.gridk_e_outside_send.procs.size() * sizeof(HYPRE_Real *));
   AllocCommVars(&(dmem_all_data->comm.gridk_e_outside_send));

   for (int i = 0; i < dmem_all_data->comm.gridk_e_outside_send.procs.size(); i++){
      dmem_all_data->comm.gridk_e_outside_send.start[i] = dmem_all_data->comm.gridk_r_outside_recv.start[i];
      dmem_all_data->comm.gridk_e_outside_send.end[i] = dmem_all_data->comm.gridk_r_outside_recv.end[i];
      dmem_all_data->comm.gridk_e_outside_send.len[i] = dmem_all_data->comm.gridk_r_outside_recv.len[i];
      dmem_all_data->comm.gridk_e_outside_send.data[i] =
         (HYPRE_Real *)calloc(dmem_all_data->comm.gridk_e_outside_send.len[i], sizeof(HYPRE_Real));
   }

/* gridk correct inside recv */
   dmem_all_data->comm.gridk_e_inside_recv.type = GRIDK_INSIDE_RECV;

   dmem_all_data->comm.gridk_e_inside_recv.procs = dmem_all_data->comm.gridk_r_inside_send.procs;
   dmem_all_data->comm.gridk_e_inside_recv.data =
      (HYPRE_Real **)malloc(dmem_all_data->comm.gridk_e_inside_recv.procs.size() * sizeof(HYPRE_Real *));
   AllocCommVars(&(dmem_all_data->comm.gridk_e_inside_recv));

   for (int i = 0; i < dmem_all_data->comm.gridk_e_inside_recv.procs.size(); i++){
      dmem_all_data->comm.gridk_e_inside_recv.start[i] = dmem_all_data->comm.gridk_r_inside_send.start[i];
      dmem_all_data->comm.gridk_e_inside_recv.end[i] = dmem_all_data->comm.gridk_r_inside_send.end[i];
      dmem_all_data->comm.gridk_e_inside_recv.len[i] = dmem_all_data->comm.gridk_r_inside_send.len[i];
      dmem_all_data->comm.gridk_e_inside_recv.data[i] =
         (HYPRE_Real *)calloc(dmem_all_data->comm.gridk_e_inside_recv.len[i], sizeof(HYPRE_Real));
   }

/* gridk correct outside recv */
   dmem_all_data->comm.gridk_e_outside_recv.type = GRIDK_OUTSIDE_RECV;

   dmem_all_data->comm.gridk_e_outside_recv.procs = dmem_all_data->comm.gridk_r_outside_send.procs;
   dmem_all_data->comm.gridk_e_outside_recv.data =
      (HYPRE_Real **)malloc(dmem_all_data->comm.gridk_e_outside_recv.procs.size() * sizeof(HYPRE_Real *));
   AllocCommVars(&(dmem_all_data->comm.gridk_e_outside_recv));

   for (int i = 0; i < dmem_all_data->comm.gridk_e_outside_recv.procs.size(); i++){
      dmem_all_data->comm.gridk_e_outside_recv.start[i] = dmem_all_data->comm.gridk_r_outside_send.start[i];
      dmem_all_data->comm.gridk_e_outside_recv.end[i] = dmem_all_data->comm.gridk_r_outside_send.end[i];
      dmem_all_data->comm.gridk_e_outside_recv.len[i] = dmem_all_data->comm.gridk_r_outside_send.len[i];
      dmem_all_data->comm.gridk_e_outside_recv.data[i] =
         (HYPRE_Real *)calloc(dmem_all_data->comm.gridk_e_outside_recv.len[i], sizeof(HYPRE_Real));
   }

/************************************
* FINE
************************************/
   hypre_ParCSRCommPkg *comm_pkg =
      hypre_ParCSRMatrixCommPkg(dmem_all_data->matrix.A_fine);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   dmem_all_data->comm.fine_send_data =
      (HYPRE_Real *)calloc(hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), sizeof(HYPRE_Real));
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(dmem_all_data->matrix.A_fine);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
   dmem_all_data->comm.fine_recv_data =
      (HYPRE_Real *)calloc(num_cols_offd, sizeof(HYPRE_Real));
   int j;

/* fine inside send */
   dmem_all_data->comm.fine_inside_send.type = FINE_INSIDE_SEND;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 1){
         dmem_all_data->comm.fine_inside_send.procs.push_back(ip);
      }
   }
   dmem_all_data->comm.fine_inside_send.hypre_map =
      (int *)malloc(dmem_all_data->comm.fine_inside_send.procs.size() * sizeof(int));
   AllocCommVars(&(dmem_all_data->comm.fine_inside_send));
   j = 0;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 1){
         dmem_all_data->comm.fine_inside_send.start[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         dmem_all_data->comm.fine_inside_send.end[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
         dmem_all_data->comm.fine_inside_send.len[j] =
            dmem_all_data->comm.fine_inside_send.end[j] - dmem_all_data->comm.fine_inside_send.start[j];
         dmem_all_data->comm.fine_inside_send.hypre_map[j] = i; 
         j++;
      }
   }

/* fine outside send */
   dmem_all_data->comm.fine_outside_send.type = FINE_OUTSIDE_SEND;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 0){
         dmem_all_data->comm.fine_outside_send.procs.push_back(ip);
      }
   }
   dmem_all_data->comm.fine_outside_send.hypre_map =
      (int *)malloc(dmem_all_data->comm.fine_outside_send.procs.size() * sizeof(int));
   AllocCommVars(&(dmem_all_data->comm.fine_outside_send));
   j = 0;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 0){
         dmem_all_data->comm.fine_outside_send.start[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         dmem_all_data->comm.fine_outside_send.end[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
         dmem_all_data->comm.fine_outside_send.len[j] =
            dmem_all_data->comm.fine_outside_send.end[j] - dmem_all_data->comm.fine_outside_send.start[j];
         dmem_all_data->comm.fine_outside_send.hypre_map[j] = i;
         j++;
      }
   }

/* fine inside recv */
   dmem_all_data->comm.fine_inside_recv.type = FINE_INSIDE_RECV;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 1){
         dmem_all_data->comm.fine_inside_recv.procs.push_back(ip);
      }
   }
   dmem_all_data->comm.fine_inside_recv.hypre_map =
      (int *)malloc(dmem_all_data->comm.fine_inside_recv.procs.size() * sizeof(int));
   AllocCommVars(&(dmem_all_data->comm.fine_inside_recv));
   j = 0;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 1){
         dmem_all_data->comm.fine_inside_recv.start[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
         dmem_all_data->comm.fine_inside_recv.end[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i+1);
         dmem_all_data->comm.fine_inside_recv.len[j] =
            dmem_all_data->comm.fine_inside_recv.end[j] - dmem_all_data->comm.fine_inside_recv.start[j];
         dmem_all_data->comm.fine_inside_recv.hypre_map[j] = i;
         j++;
      }
   }

/* fine outside recv */
   dmem_all_data->comm.fine_outside_recv.type = FINE_OUTSIDE_RECV;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 0){
         dmem_all_data->comm.fine_outside_recv.procs.push_back(ip);
      }
   }
   dmem_all_data->comm.fine_outside_recv.hypre_map =
      (int *)malloc(dmem_all_data->comm.fine_outside_recv.procs.size() * sizeof(int));
   AllocCommVars(&(dmem_all_data->comm.fine_outside_recv));
   j = 0;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 0){
         dmem_all_data->comm.fine_outside_recv.start[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
         dmem_all_data->comm.fine_outside_recv.end[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i+1);
         dmem_all_data->comm.fine_outside_recv.len[j] =
            dmem_all_data->comm.fine_outside_recv.end[j] - dmem_all_data->comm.fine_outside_recv.start[j];
         dmem_all_data->comm.fine_outside_recv.hypre_map[j] = i;
         j++;
      }
   }

  // if (my_id == 0) printf("inside send:\n");
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("\t%d, %d:", p, my_grid);
  //       for (int i = 0; i < dmem_all_data->comm.gridk_r_inside_send.procs.size(); i++){
  //          printf(" (%d,%d,%d)",
  //                 dmem_all_data->comm.gridk_r_inside_send.start[i],
  //                 dmem_all_data->comm.gridk_r_inside_send.end[i],
  //                 hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(dmem_all_data->matrix.A_fine)));
  //       }
  //       printf("\n");
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
  // if (my_id == 0) printf("inside recv:\n");
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("\t%d, %d:", p, my_grid);
  //       for (int i = 0; i < dmem_all_data->comm.gridk_r_inside_recv.procs.size(); i++){
  //          printf(" (%d,%d,%d)", 
  //                 dmem_all_data->comm.gridk_r_inside_recv.start[i],
  //                 dmem_all_data->comm.gridk_r_inside_recv.end[i],
  //                 hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(dmem_all_data->matrix.A_gridk)));
  //       }
  //       printf("\n");
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }

  // if (my_id == 0) printf("\n\noutside send:\n");
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("\t%d, %d:", p, my_grid);
  //       for (int i = 0; i < dmem_all_data->comm.gridk_e_outside_send.procs.size(); i++){
  //          printf(" %d", dmem_all_data->comm.gridk_e_outside_send.procs[i]);
  //       }
  //       printf("\n");
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
  // if (my_id == 0) printf("outside recv:\n");
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("\t%d, %d:", p, my_grid);
  //       for (int i = 0; i < dmem_all_data->comm.gridk_e_outside_recv.procs.size(); i++){
  //          printf(" %d", dmem_all_data->comm.gridk_e_outside_recv.procs[i]);
  //       }
  //       printf("\n");
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void ConstructVectors(DMEM_AllData *dmem_all_data,
                 hypre_ParCSRMatrix *A,
                 DMEM_VectorData *vector)
{
   void *object;
   HYPRE_Real *values;
   HYPRE_IJVector ij_x = NULL;
   HYPRE_IJVector ij_b = NULL;
   HYPRE_IJVector ij_r = NULL;
   HYPRE_IJVector ij_e = NULL;

   HYPRE_Int first_local_row, last_local_row;
   HYPRE_Int first_local_col, last_local_col;
   HYPRE_ParCSRMatrixGetLocalRange(A,
                                   &first_local_row, &last_local_row ,
                                   &first_local_col, &last_local_col );
   HYPRE_Int local_num_rows = last_local_row - first_local_row + 1;
   HYPRE_Int local_num_cols = last_local_col - first_local_col + 1;

   /* initialize fine grid approximation to the solution */
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
   HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_x);

   values = hypre_CTAlloc(HYPRE_Real, local_num_cols, HYPRE_MEMORY_HOST);
   for (HYPRE_Int i = 0; i < local_num_cols; i++) values[i] = 0.0;
   HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
   hypre_TFree(values, HYPRE_MEMORY_HOST);

   HYPRE_IJVectorGetObject(ij_x, &object);
   vector->x = (HYPRE_ParVector)object;

   /* initialize correction vector */
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_e);
   HYPRE_IJVectorSetObjectType(ij_e, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_e);

   values = hypre_CTAlloc(HYPRE_Real, local_num_cols, HYPRE_MEMORY_HOST);
   HYPRE_IJVectorSetValues(ij_e, local_num_cols, NULL, values);
   hypre_TFree(values, HYPRE_MEMORY_HOST);

   HYPRE_IJVectorGetObject(ij_e, &object);
   vector->e = (HYPRE_ParVector)object;

   /* initialize fine grid right-hand side */
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
   HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_b);

   values = hypre_CTAlloc(HYPRE_Real, local_num_rows, HYPRE_MEMORY_HOST);
   for (HYPRE_Int i = 0; i < local_num_rows; i++) values[i] = 1.0;
   HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);

   HYPRE_IJVectorGetObject(ij_b, &object);
   vector->b = (HYPRE_ParVector)object;

   /* initialize fine grid residual */
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_r);
   HYPRE_IJVectorSetObjectType(ij_r, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_r);

   HYPRE_IJVectorSetValues(ij_r, local_num_rows, NULL, values);
   hypre_TFree(values, HYPRE_MEMORY_HOST);

   HYPRE_IJVectorGetObject(ij_r, &object);
   vector->r = (HYPRE_ParVector)object;
}

void GridkResetCommData(DMEM_CommData *comm_data)
{
   for (int i = 0; i < comm_data->procs.size(); i++){
      comm_data->requests[i] = MPI_REQUEST_NULL;
      for (int j = 0; j < comm_data->len[i]; j++){
         comm_data->data[i][j] = 0;
      }
   }
}

void ResetVector(DMEM_AllData *dmem_all_data,
                 DMEM_VectorData *vector)
{
   hypre_ParVector *x = vector->x;
   hypre_ParVector *b = vector->b;
   hypre_ParVector *r = vector->r;
   hypre_ParVector *e = vector->e;
   hypre_Vector *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector *b_local  = hypre_ParVectorLocalVector(b);
   hypre_Vector *r_local  = hypre_ParVectorLocalVector(r);
   hypre_Vector *e_local  = hypre_ParVectorLocalVector(e);
   HYPRE_Real *x_local_data = hypre_VectorData(x_local);
   HYPRE_Real *b_local_data = hypre_VectorData(b_local);
   HYPRE_Real *r_local_data = hypre_VectorData(r_local);
   HYPRE_Real *e_local_data = hypre_VectorData(e_local);
   for (int i = 0; i < hypre_VectorSize(x_local); i++){
      x_local_data[i] = 0.0;
     // b_local_data[i] = r_local_data[i] = RandDouble(-1.0,1.0);
      b_local_data[i] = r_local_data[i] = 1.0;
      e_local_data[i] = 0.0;
   }
}

void DMEM_ResetData(DMEM_AllData *dmem_all_data)
{
   GridkResetCommData(&(dmem_all_data->comm.gridk_r_inside_send));
   GridkResetCommData(&(dmem_all_data->comm.gridk_r_inside_recv));
   GridkResetCommData(&(dmem_all_data->comm.gridk_r_outside_send));
   GridkResetCommData(&(dmem_all_data->comm.gridk_r_outside_recv));
   GridkResetCommData(&(dmem_all_data->comm.gridk_e_inside_send));
   GridkResetCommData(&(dmem_all_data->comm.gridk_e_inside_recv));
   GridkResetCommData(&(dmem_all_data->comm.gridk_e_outside_send));
   GridkResetCommData(&(dmem_all_data->comm.gridk_e_outside_recv));


   for (int i = 0; i < dmem_all_data->comm.fine_inside_send.procs.size(); i++){
      dmem_all_data->comm.fine_inside_send.requests[i] = MPI_REQUEST_NULL;
   }
   for (int i = 0; i < dmem_all_data->comm.fine_inside_recv.procs.size(); i++){
      dmem_all_data->comm.fine_outside_send.requests[i] = MPI_REQUEST_NULL;
   }

   for (int i = 0; i < dmem_all_data->comm.fine_outside_send.procs.size(); i++){
      dmem_all_data->comm.fine_outside_send.requests[i] = MPI_REQUEST_NULL;
   }
   for (int i = 0; i < dmem_all_data->comm.fine_outside_recv.procs.size(); i++){
      dmem_all_data->comm.fine_outside_recv.requests[i] = MPI_REQUEST_NULL;
   }

   hypre_ParCSRMatrix *A = dmem_all_data->matrix.A_fine;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);

   ResetVector(dmem_all_data, &(dmem_all_data->vector_fine));
   ResetVector(dmem_all_data, &(dmem_all_data->vector_gridk));

   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   for (int i = 0; i < hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends); i++){
      dmem_all_data->comm.fine_send_data[i] = 0;
   }
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
   for (int i = 0; i < num_cols_offd; i++){
      dmem_all_data->comm.fine_recv_data[i] = 0;
   }
}

void PartitionProcs(DMEM_AllData *dmem_all_data)
{
   int num_levels = dmem_all_data->grid.num_levels;
   int num_procs, my_id;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   dmem_all_data->grid.procs = (int **)calloc(num_levels, sizeof(int *));
   dmem_all_data->grid.num_procs_level = 
      (int *)calloc(num_levels, sizeof(int));

   int procs_per_level = num_procs/num_levels;
   int extra = num_procs - procs_per_level*(num_levels - 1);
   for (int level = 0; level < num_levels-1; level++){
      dmem_all_data->grid.num_procs_level[level] = procs_per_level;
   }
   dmem_all_data->grid.num_procs_level[num_levels-1] = extra;

   
   dmem_all_data->grid.my_grid = -1;
   int count_id = 0;
   for (int level = 0; level < num_levels; level++){
      dmem_all_data->grid.procs[level] = 
         (int *)calloc(dmem_all_data->grid.num_procs_level[level], sizeof(int));
      for (int i = 0; i < dmem_all_data->grid.num_procs_level[level]; i++){
         dmem_all_data->grid.procs[level][i] = count_id;
	 if (my_id == count_id){
	    dmem_all_data->grid.my_grid = level;
	 }
	 count_id++;
      }
   }
   MPI_Comm_split(MPI_COMM_WORLD,
                  dmem_all_data->grid.my_grid,
                  my_id,
                  &(dmem_all_data->grid.my_comm));
}

void ComputeWork(DMEM_AllData *dmem_all_data)
{
}

void ConstructMatrix(DMEM_AllData *dmem_all_data,
		     HYPRE_ParCSRMatrix *A,
		     MPI_Comm comm)
{
   DMEM_Laplacian_3D_27pt(dmem_all_data,
                          A,
                          comm,
                          dmem_all_data->matrix.nx,
                          dmem_all_data->matrix.ny,
                          dmem_all_data->matrix.nz);
}
