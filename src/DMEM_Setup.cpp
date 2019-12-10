#include "Main.hpp"
#include "Misc.hpp"
#include "DMEM_Setup.hpp"
#include "DMEM_Main.hpp"
#include "DMEM_BuildMatrix.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_ParMfem.hpp"
#include "DMEM_Add.hpp"
#include "DMEM_Comm.hpp"
#include "DMEM_Misc.hpp"

using namespace std;

void SetHypreSolver(DMEM_AllData *dmem_all_data,
                    HYPRE_Solver *solver);
void SetMultaddHypreSolver(DMEM_AllData *dmem_all_data,
                           HYPRE_Solver *solver);
void ConstructVectors(DMEM_AllData *dmem_all_data,
                      hypre_ParCSRMatrix *A,
                      DMEM_VectorData *vector,
                      int fine_flag);
void ComputeWork(DMEM_AllData *all_data);
void AssignProcs(DMEM_AllData *dmem_all_data);
void PartitionGrids(DMEM_AllData *dmem_all_data);
void BuildMatrix(DMEM_AllData *dmem_all_data, HYPRE_ParCSRMatrix *A, HYPRE_ParVector *rhs, MPI_Comm comm);
void CreateCommData_GlobalRes(DMEM_AllData *dmem_all_data);
void CreateCommData_LocalRes(DMEM_AllData *dmem_all_data);
void SetVectorComms(DMEM_AllData *dmem_all_data,
                    DMEM_VectorData *vector,
                    MPI_Comm comm);
void DistributeMatrix(DMEM_AllData *dmem_all_data,
                      hypre_ParCSRMatrix *A,
                      hypre_ParCSRMatrix **B);
void ComputeWork(AllData *dmem_all_data);

//TODO: clear extra memory
void DMEM_Setup(DMEM_AllData *dmem_all_data)
{
   hypre_ParAMGData *amg_data;
   hypre_ParAMGData *amg_data_fine, *amg_data_gridk;
   hypre_ParCSRMatrix **A_array, **P_array, **R_array;
   double start;
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
   amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;

   start = omp_get_wtime();
   /* fine */
   BuildMatrix(dmem_all_data, &(dmem_all_data->matrix.A_fine), &(dmem_all_data->vector_fine.b), MPI_COMM_WORLD);
  // char buffer[100];
  // sprintf(buffer, "A_%d.txt", num_procs);
  // DMEM_PrintParCSRMatrix(dmem_all_data->matrix.A_fine, buffer);
   ConstructVectors(dmem_all_data,
                    dmem_all_data->matrix.A_fine,
                    &(dmem_all_data->vector_fine), 1);
   if (dmem_all_data->input.solver == MULT_MULTADD ||
       dmem_all_data->input.solver == MULT ||
       dmem_all_data->input.solver == BPX ||
       dmem_all_data->input.solver == AFACX ||
       dmem_all_data->input.solver == SYNC_AFACX /*||
       dmem_all_data->input.solver == SYNC_MULTADD*/){
      SetHypreSolver(dmem_all_data,
                     &(dmem_all_data->hypre.solver));
   }
   else {
      SetMultaddHypreSolver(dmem_all_data,
                            &(dmem_all_data->hypre.solver));
   }
   amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   if (dmem_all_data->input.num_interpolants == NUMLEVELS_INTERPOLANTS){
      if (dmem_all_data->input.P_gridk_droptol_flag == 1){
         hypre_ParAMGDataMultAddTruncFactor(amg_data_fine) = dmem_all_data->input.P_gridk_droptol;
      }
      if (dmem_all_data->input.P_gridk_maxelmts_flag == 1){
         hypre_ParAMGDataMultAddPMaxElmts(amg_data_fine) = dmem_all_data->input.P_gridk_maxelmts;
      }
   }
   HYPRE_BoomerAMGSetup(dmem_all_data->hypre.solver,
			dmem_all_data->matrix.A_fine,
			dmem_all_data->vector_fine.f,
			dmem_all_data->vector_fine.u);
  // HYPRE_BoomerAMGSetMaxIter(dmem_all_data->hypre.solver, dmem_all_data->input.num_cycles);
  // HYPRE_BoomerAMGSolve(dmem_all_data->hypre.solver,
  //                      dmem_all_data->matrix.A_fine,
  //                      dmem_all_data->vector_fine.b,
  //                      dmem_all_data->vector_fine.x);
   amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   dmem_all_data->grid.num_levels = hypre_ParAMGDataNumLevels(amg_data_fine);
  // int temp_num_levels = hypre_ParAMGDataNumLevels(amg_data_fine);
  // for (int level = 0; level < temp_num_levels; level++){
  //    int participant = hypre_ParCSRMatrixNumRows(hypre_ParAMGDataAArray(amg_data_fine)[level]);
  //    int min_participant;
  //    MPI_Allreduce(&participant,
  //                  &min_participant,
  //                  1,
  //                  MPI_INT,
  //                  MPI_MIN,
  //                  MPI_COMM_WORLD);
  //    if (min_participant == 0) break;
  //    dmem_all_data->grid.num_levels = level+1;
  // }
  // hypre_ParAMGDataNumLevels(amg_data_fine) = dmem_all_data->grid.num_levels;
   dmem_all_data->input.coarsest_mult_level = min(dmem_all_data->input.coarsest_mult_level, dmem_all_data->grid.num_levels-1);

   if (dmem_all_data->input.solver == MULTADD ||
       dmem_all_data->input.solver == BPX ||
       dmem_all_data->input.solver == MULT_MULTADD ||
       dmem_all_data->input.solver == AFACX){
      if (dmem_all_data->input.num_interpolants == ONE_INTERPOLANT){
         P_array = hypre_ParAMGDataPArray(amg_data_fine);
         dmem_all_data->matrix.P_fine = (hypre_ParCSRMatrix **)malloc(dmem_all_data->grid.num_levels * sizeof(hypre_ParCSRMatrix *));
         dmem_all_data->matrix.P_fine[0] = P_array[0];
         hypre_ParCSRMatrixSetNumNonzeros(dmem_all_data->matrix.P_fine[0]);
         for (int level = 0; level < dmem_all_data->grid.num_levels-2; level++){
            dmem_all_data->matrix.P_fine[level+1] = NULL;
            dmem_all_data->matrix.P_fine[level+1] = hypre_ParMatmul(dmem_all_data->matrix.P_fine[level], P_array[level+1]);
            hypre_ParCSRMatrixOwnsColStarts(dmem_all_data->matrix.P_fine[level+1]) = 0;
            hypre_ParCSRMatrixSetNumNonzeros(dmem_all_data->matrix.P_fine[level+1]);
            hypre_MatvecCommPkgCreate(dmem_all_data->matrix.P_fine[level+1]);
           // if (dmem_all_data->input.P_gridk_droptol_flag == 1 || dmem_all_data->input.P_gridk_maxelmts_flag == 1){
           //    hypre_BoomerAMGTruncandBuild(dmem_all_data->matrix.P_fine[level+1], dmem_all_data->input.P_gridk_droptol, dmem_all_data->input.P_gridk_maxelmts);
           // }
         }
        // for (int level = 0; level < dmem_all_data->grid.num_levels-1; level++){
        //    char filename[100];
        //    sprintf(filename, "P_fine_%d", level+1);
        //    DMEM_PrintParCSRMatrix(dmem_all_data->matrix.P_fine[level], filename);
        // }
      } 
      ComputeWork(dmem_all_data);
      AssignProcs(dmem_all_data);
      DistributeMatrix(dmem_all_data,
                       dmem_all_data->matrix.A_fine,
                       &(dmem_all_data->matrix.A_gridk));
      hypre_MatvecCommPkgCreate(dmem_all_data->matrix.A_gridk);
      ConstructVectors(dmem_all_data,
                       dmem_all_data->matrix.A_gridk,
                       &(dmem_all_data->vector_gridk), 0);
      if (dmem_all_data->input.solver == BPX ||
          dmem_all_data->input.solver == AFACX){
         SetHypreSolver(dmem_all_data,
                        &(dmem_all_data->hypre.solver_gridk));
      }
      else {
         SetMultaddHypreSolver(dmem_all_data,
                               &(dmem_all_data->hypre.solver_gridk));
         HYPRE_BoomerAMGSetMaxLevels(dmem_all_data->hypre.solver_gridk, dmem_all_data->grid.num_levels);
      }
      amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
      hypre_ParAMGDataNumLevels(amg_data_gridk) = dmem_all_data->grid.num_levels;

      hypre_ParCSRMatrix **A_array_fine = hypre_ParAMGDataAArray(amg_data_fine);
      hypre_ParCSRMatrix **P_array_fine = hypre_ParAMGDataPArray(amg_data_fine);
     // hypre_ParCSRMatrix *Q;
      hypre_ParCSRMatrix **A_array_gridk = hypre_ParAMGDataAArray(amg_data_gridk);
      A_array_gridk = hypre_CTAlloc(hypre_ParCSRMatrix *, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
      for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
         DistributeMatrix(dmem_all_data,
                          A_array_fine[level],
                          &(A_array_gridk[level]));
         hypre_ParCSRMatrixOwnsRowStarts(A_array_gridk[level]) = 1;
         hypre_ParCSRMatrixOwnsColStarts(A_array_gridk[level]) = 0;
         hypre_MatvecCommPkgCreate(A_array_gridk[level]);
        // for (int p = 0; p < num_procs; p++){
        //    if (my_id == p && dmem_all_data->grid.my_grid == 0){
        //       printf("%d %d %d %d\n", my_id, level, hypre_ParCSRMatrixFirstRowIndex(A_array_fine[level]), hypre_ParCSRMatrixLastRowIndex(A_array_fine[level]));
        //    }
        //    MPI_Barrier(MPI_COMM_WORLD);
        // }
      }
      hypre_ParAMGDataAArray(amg_data_gridk) = A_array_gridk;

      if (dmem_all_data->input.num_interpolants == ONE_INTERPOLANT){
         for (int level = 1; level < dmem_all_data->grid.num_levels; level++){
            if (dmem_all_data->grid.my_grid == level){ 
               DistributeMatrix(dmem_all_data,
                                dmem_all_data->matrix.P_fine[level-1],
                                &(dmem_all_data->matrix.P_gridk));
               hypre_ParCSRMatrixOwnsColStarts(dmem_all_data->matrix.P_gridk) = 0;
               hypre_MatvecCommPkgCreate(dmem_all_data->matrix.P_gridk);
               dmem_all_data->matrix.R_gridk = dmem_all_data->matrix.P_gridk;
            }
            else {
               hypre_ParCSRMatrix *Q;
               DistributeMatrix(dmem_all_data,
                                dmem_all_data->matrix.P_fine[level-1],
                                &Q);
            }
         }

        // int my_id_gridk; 
        // MPI_Comm_rank(dmem_all_data->grid.my_comm, &my_id_gridk);
        // hypre_ParCSRMatrixSetNumNonzeros(dmem_all_data->matrix.P_gridk);
        // char filename[100];
        // sprintf(filename, "P_gridk_%d", dmem_all_data->grid.my_grid);
        // if (dmem_all_data->grid.my_grid > 1) printf("id %d: %d %d\n", my_id, hypre_ParCSRMatrixNumNonzeros(dmem_all_data->matrix.P_fine[dmem_all_data->grid.my_grid-1]), hypre_ParCSRMatrixNumNonzeros(dmem_all_data->matrix.P_gridk));
        // }
      }
      else {
         hypre_ParCSRMatrix **P_array_gridk = hypre_ParAMGDataPArray(amg_data_gridk);
         P_array_gridk = hypre_CTAlloc(hypre_ParCSRMatrix *, dmem_all_data->grid.num_levels-1, dmem_all_data->input.hypre_memory);
         for (int level = 0; level < dmem_all_data->grid.num_levels-1; level++){
            DistributeMatrix(dmem_all_data,
                             P_array_fine[level],
                             &(P_array_gridk[level]));
            hypre_ParCSRMatrixOwnsColStarts(P_array_gridk[level]) = 0;
            hypre_MatvecCommPkgCreate(P_array_gridk[level]);
            hypre_ParCSRMatrixSetNumNonzeros(P_array_gridk[level]);
            hypre_ParCSRMatrixSetNumNonzeros(P_array_fine[level]);
           // if (my_id == 0)
           // printf("level %d: %d %d\n", level, hypre_ParCSRMatrixNumNonzeros(P_array_gridk[level]), hypre_ParCSRMatrixNumNonzeros(P_array_fine[level]));
         }
         hypre_ParAMGDataPArray(amg_data_gridk) = P_array_gridk;
         hypre_ParAMGDataRArray(amg_data_gridk) = P_array_gridk;
      }

      hypre_ParVector *Vtemp_gridk = hypre_ParAMGDataVtemp(amg_data_gridk);
      Vtemp_gridk = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array_gridk[0]),
                                          hypre_ParCSRMatrixGlobalNumRows(A_array_gridk[0]),
                                          hypre_ParCSRMatrixRowStarts(A_array_gridk[0]));
      hypre_ParVectorInitialize(Vtemp_gridk);
      hypre_ParVectorSetPartitioningOwner(Vtemp_gridk, 0);
      hypre_ParAMGDataVtemp(amg_data_gridk) = Vtemp_gridk;

      hypre_ParVector **F_array_gridk = hypre_ParAMGDataFArray(amg_data_gridk);
      hypre_ParVector **U_array_gridk = hypre_ParAMGDataUArray(amg_data_gridk);

      F_array_gridk = hypre_CTAlloc(hypre_ParVector*, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
      U_array_gridk = hypre_CTAlloc(hypre_ParVector*, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
      
      F_array_gridk[0] = dmem_all_data->vector_gridk.f;
      U_array_gridk[0] = dmem_all_data->vector_gridk.u;
      for (int level = 1; level < dmem_all_data->grid.num_levels; level++){
         F_array_gridk[level] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array_gridk[level]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_array_gridk[level]),
                                  hypre_ParCSRMatrixRowStarts(A_array_gridk[level]));
         hypre_ParVectorInitialize(F_array_gridk[level]);
         hypre_ParVectorSetPartitioningOwner(F_array_gridk[level], 0);

         U_array_gridk[level] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array_gridk[level]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_array_gridk[level]),
                                  hypre_ParCSRMatrixRowStarts(A_array_gridk[level]));
         hypre_ParVectorInitialize(U_array_gridk[level]);
         hypre_ParVectorSetPartitioningOwner(U_array_gridk[level], 0);
      }
      hypre_ParAMGDataFArray(amg_data_gridk) = F_array_gridk;
      hypre_ParAMGDataUArray(amg_data_gridk) = U_array_gridk;

     // if (dmem_all_data->input.num_interpolants == NUMLEVELS_INTERPOLANTS){
     //    if (dmem_all_data->input.P_gridk_droptol_flag == 1){
     //       hypre_ParAMGDataMultAddTruncFactor(amg_data_gridk) = dmem_all_data->input.P_gridk_droptol;
     //    }
     //    if (dmem_all_data->input.P_gridk_maxelmts_flag == 1){
     //       hypre_ParAMGDataMultAddPMaxElmts(amg_data_gridk) = dmem_all_data->input.P_gridk_maxelmts; 
     //    }
     // }
     // HYPRE_BoomerAMGSetup(dmem_all_data->hypre.solver_gridk,
     //                      dmem_all_data->matrix.A_gridk,
     //                      dmem_all_data->vector_gridk.f,
     //                      dmem_all_data->vector_gridk.u);
      if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
         CreateCommData_GlobalRes(dmem_all_data);
      }
      else {
         CreateCommData_LocalRes(dmem_all_data);
      }
   }

   if (dmem_all_data->input.solver == MULT ||
       dmem_all_data->input.solver == SYNC_MULTADD ||
       dmem_all_data->input.solver == SYNC_AFACX){
      hypre_GaussElimSetup(amg_data_fine, hypre_ParAMGDataNumLevels(amg_data_fine)-1, 9);
   }
   else {
      hypre_GaussElimSetup(amg_data_gridk, dmem_all_data->grid.num_levels-1, 9);
   }

   if (dmem_all_data->input.smoother == L1_JACOBI){
      if (dmem_all_data->input.solver == MULT_MULTADD ||
          dmem_all_data->input.solver == MULT ||
          dmem_all_data->input.solver == SYNC_MULTADD ||
          dmem_all_data->input.solver == SYNC_AFACX){
         dmem_all_data->matrix.L1_row_norm_fine = hypre_CTAlloc(HYPRE_Real *, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
         A_array = hypre_ParAMGDataAArray(amg_data_fine);
         for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
            HYPRE_Real *A_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[level]));
            HYPRE_Int *A_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[level]));
            HYPRE_Real *A_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A_array[level]));
            HYPRE_Int *A_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A_array[level]));
            int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
            dmem_all_data->matrix.L1_row_norm_fine[level] = hypre_CTAlloc(HYPRE_Real, num_rows, dmem_all_data->input.hypre_memory);
            for (int i = 0; i < num_rows; i++){
               dmem_all_data->matrix.L1_row_norm_fine[level][i] = A_diag_data[A_diag_i[i]];
               for (int jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++){
                  dmem_all_data->matrix.L1_row_norm_fine[level][i] += fabs(A_diag_data[jj]);
               }
               for (int jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++){
                  dmem_all_data->matrix.L1_row_norm_fine[level][i] += fabs(A_offd_data[jj]);
               }
            }
         }
      }
      if (dmem_all_data->input.solver == MULTADD ||
          dmem_all_data->input.solver == BPX ||
          dmem_all_data->input.solver == MULT_MULTADD ||
          dmem_all_data->input.solver == AFACX){
         dmem_all_data->matrix.L1_row_norm_gridk = hypre_CTAlloc(HYPRE_Real *, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
         dmem_all_data->matrix.symmL1_row_norm_gridk = hypre_CTAlloc(HYPRE_Real *, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
         A_array = hypre_ParAMGDataAArray(amg_data_gridk);
         for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
            HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[level]));
            HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[level]));
            int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
            dmem_all_data->matrix.L1_row_norm_gridk[level] = hypre_CTAlloc(HYPRE_Real, num_rows, dmem_all_data->input.hypre_memory);
            dmem_all_data->matrix.symmL1_row_norm_gridk[level] = hypre_CTAlloc(HYPRE_Real, num_rows, dmem_all_data->input.hypre_memory);
            for (int i = 0; i < num_rows; i++){
               dmem_all_data->matrix.L1_row_norm_gridk[level][i] = 0;
               for (int jj = A_i[i]; jj < A_i[i+1]; jj++){
                  dmem_all_data->matrix.L1_row_norm_gridk[level][i] += fabs(A_data[jj]);
               }
               dmem_all_data->matrix.symmL1_row_norm_gridk[level][i] = -dmem_all_data->matrix.L1_row_norm_gridk[level][i] / 2.0;
            }
         }
      }
   }
   else if (dmem_all_data->input.smoother == JACOBI ||
            dmem_all_data->input.smoother == ASYNC_JACOBI){
      if (dmem_all_data->input.solver == MULT_MULTADD ||
          dmem_all_data->input.solver == MULT ||
          dmem_all_data->input.solver == SYNC_MULTADD ||
          dmem_all_data->input.solver == SYNC_AFACX){
         dmem_all_data->matrix.wJacobi_scale_fine = hypre_CTAlloc(HYPRE_Real *, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
         A_array = hypre_ParAMGDataAArray(amg_data_fine);
         for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
            HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[level]));
            HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[level]));
            int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
            dmem_all_data->matrix.wJacobi_scale_fine[level] = hypre_CTAlloc(HYPRE_Real, num_rows, dmem_all_data->input.hypre_memory);
            for (int i = 0; i < num_rows; i++){
               dmem_all_data->matrix.wJacobi_scale_fine[level][i] = A_data[A_i[i]] / dmem_all_data->input.smooth_weight;
            }
         }
      }
      if (dmem_all_data->input.solver == MULTADD ||
          dmem_all_data->input.solver == BPX ||
          dmem_all_data->input.solver == MULT_MULTADD ||
          dmem_all_data->input.solver == AFACX){
         dmem_all_data->matrix.wJacobi_scale_gridk = hypre_CTAlloc(HYPRE_Real *, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
         dmem_all_data->matrix.symmwJacobi_scale_gridk = hypre_CTAlloc(HYPRE_Real *, dmem_all_data->grid.num_levels, dmem_all_data->input.hypre_memory);
         A_array = hypre_ParAMGDataAArray(amg_data_gridk);
         for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
            HYPRE_Real *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_array[level]));
            HYPRE_Int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_array[level]));
            int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
            dmem_all_data->matrix.wJacobi_scale_gridk[level] = hypre_CTAlloc(HYPRE_Real, num_rows, dmem_all_data->input.hypre_memory);
            dmem_all_data->matrix.symmwJacobi_scale_gridk[level] = hypre_CTAlloc(HYPRE_Real, num_rows, dmem_all_data->input.hypre_memory);
            for (int i = 0; i < num_rows; i++){
               dmem_all_data->matrix.wJacobi_scale_gridk[level][i] = A_data[A_i[i]] / dmem_all_data->input.smooth_weight;
               dmem_all_data->matrix.symmwJacobi_scale_gridk[level][i] = -2.0 * dmem_all_data->matrix.wJacobi_scale_gridk[level][i];
            }
         }
      }
   }
}

void SetHypreSolver(DMEM_AllData *dmem_all_data,
		    HYPRE_Solver *solver)
{
   HYPRE_BoomerAMGCreate(solver);
   HYPRE_BoomerAMGSetPrintLevel(*solver, dmem_all_data->hypre.print_level);
   HYPRE_BoomerAMGSetMaxRowSum(*solver, 1.0);
  // HYPRE_BoomerAMGSetOldDefault(*solver);
   HYPRE_BoomerAMGSetPostInterpType(*solver, 0);
   HYPRE_BoomerAMGSetInterpType(*solver, dmem_all_data->hypre.interp_type);
   HYPRE_BoomerAMGSetRestriction(*solver, 0);
   HYPRE_BoomerAMGSetCoarsenType(*solver, dmem_all_data->hypre.coarsen_type);
   HYPRE_BoomerAMGSetMaxLevels(*solver, dmem_all_data->hypre.max_levels);
   HYPRE_BoomerAMGSetAggNumLevels(*solver, dmem_all_data->hypre.agg_num_levels);
   //HYPRE_BoomerAMGSetRelaxWt(*solver, dmem_all_data->input.smooth_weight);
   HYPRE_BoomerAMGSetRelaxType(*solver, 18);
   HYPRE_BoomerAMGSetStrongThreshold(*solver, dmem_all_data->hypre.strong_threshold);
  // HYPRE_BoomerAMGSetPMaxElmts(*solver, 4);
   HYPRE_BoomerAMGSetNumFunctions(*solver, dmem_all_data->hypre.num_functions);
   HYPRE_BoomerAMGSetMeasureType(*solver, 1);
//   HYPRE_BoomerAMGSetCycleRelaxType(*solver, 99, 3);
}

void SetMultaddHypreSolver(DMEM_AllData *dmem_all_data,
                           HYPRE_Solver *solver)
{
   int my_grid = dmem_all_data->grid.my_grid;
   HYPRE_BoomerAMGCreate(solver);
   if (my_grid == 0){
      HYPRE_BoomerAMGSetPrintLevel(*solver, dmem_all_data->hypre.print_level);
   }
   else {
      HYPRE_BoomerAMGSetPrintLevel(*solver, 0);
   }
  // HYPRE_BoomerAMGSetOldDefault(*solver);
   HYPRE_BoomerAMGSetPostInterpType(*solver, 0);
   HYPRE_BoomerAMGSetInterpType(*solver, dmem_all_data->hypre.interp_type);
   HYPRE_BoomerAMGSetRestriction(*solver, 0);
   HYPRE_BoomerAMGSetCoarsenType(*solver, dmem_all_data->hypre.coarsen_type);
   HYPRE_BoomerAMGSetMaxLevels(*solver, dmem_all_data->hypre.max_levels);
   HYPRE_BoomerAMGSetAggNumLevels(*solver, dmem_all_data->hypre.agg_num_levels);
   if (dmem_all_data->input.smoother == L1_JACOBI){
      HYPRE_BoomerAMGSetRelaxType(*solver, 18);
   }
   else {
      HYPRE_BoomerAMGSetRelaxType(*solver, 0);
   }
   HYPRE_BoomerAMGSetRelaxWt(*solver, dmem_all_data->input.smooth_weight);
  // HYPRE_BoomerAMGSetMeasureType(*solver, 1);
//   HYPRE_BoomerAMGSetCycleRelaxType(*solver, 99, 3);

   /* multadd options */
   HYPRE_BoomerAMGSetMultAdditive(*solver, 0);
   HYPRE_BoomerAMGSetMultAddTruncFactor(*solver, dmem_all_data->hypre.multadd_trunc_factor);
   HYPRE_BoomerAMGSetAddRelaxType(*solver, 0);
   HYPRE_BoomerAMGSetAddRelaxWt(*solver, dmem_all_data->input.smooth_weight);
  //HYPRE_BoomerAMGSetMultAddPMaxElmts(*solver, hypre_ParCSRMatrixNumRows(dmem_all_data->matrix.A_fine));
   
  // hypre_BoomerAMGSetMultAdditive(*solver, dmem_all_data->input.coarsest_mult_level);

   if (dmem_all_data->input.multadd_smooth_interp_level_type == SMOOTH_INTERP_MY_GRID){
      dmem_all_data->hypre.start_smooth_level = dmem_all_data->grid.my_grid-1 - dmem_all_data->input.afacj_level;
      dmem_all_data->hypre.start_smooth_level = max(0, dmem_all_data->hypre.start_smooth_level);
      hypre_BoomerAMGSetMultAdditive(*solver, dmem_all_data->hypre.start_smooth_level);
   }

   HYPRE_BoomerAMGSetNumFunctions(*solver, dmem_all_data->hypre.num_functions);
   HYPRE_BoomerAMGSetMeasureType(*solver, 1);
}

void AllocCommVars(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   comm_data->len.resize(comm_data->procs.size());
   comm_data->start.resize(comm_data->procs.size());
   comm_data->end.resize(comm_data->procs.size());
   comm_data->message_count.resize(comm_data->procs.size());
   comm_data->done_flags.resize(comm_data->procs.size());
   comm_data->recv_flags.resize(comm_data->procs.size());
   comm_data->update_res_in_comm = 0;
   
   if (comm_data->type == GRIDK_OUTSIDE_SEND || comm_data->type == FINE_INTRA_OUTSIDE_SEND){
      comm_data->max_inflight.resize(comm_data->procs.size());
      comm_data->num_inflight.resize(comm_data->procs.size());
      comm_data->next_inflight.resize(comm_data->procs.size());
      comm_data->requests_inflight = (MPI_Request **)malloc(comm_data->procs.size() * sizeof(MPI_Request *));
      comm_data->inflight_flags = (int **)malloc(comm_data->procs.size() * sizeof(int *));
      for (int i = 0; i < comm_data->procs.size(); i++){
         comm_data->max_inflight[i] = dmem_all_data->input.max_inflight;
         comm_data->requests_inflight[i] = (MPI_Request *)malloc(comm_data->max_inflight[i] * sizeof(MPI_Request));
         comm_data->inflight_flags[i] = (int *)malloc(comm_data->max_inflight[i] * sizeof(int));
      }
   }
   
   comm_data->requests = (MPI_Request *)malloc(comm_data->procs.size() * sizeof(MPI_Request));

   if (dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL &&
       comm_data->type == FINE_INTRA_OUTSIDE_RECV &&
       dmem_all_data->grid.my_grid == 0){
      comm_data->r_norm.resize(comm_data->procs.size(), 0.0);
      comm_data->r_norm_boundary.resize(comm_data->procs.size(), 0.0);
      comm_data->r_norm_boundary_prev.resize(comm_data->procs.size(), 0.0);
   }
}

void AllocCommData(DMEM_AllData *dmem_all_data, DMEM_CommData *comm_data)
{
   if (comm_data->type == GRIDK_OUTSIDE_SEND || comm_data->type == FINE_INTRA_OUTSIDE_SEND){
      comm_data->data_inflight = hypre_CTAlloc(HYPRE_Real **, comm_data->procs.size(), dmem_all_data->input.hypre_memory);
      for (int i = 0; i < comm_data->procs.size(); i++){
         comm_data->data_inflight[i] = hypre_CTAlloc(HYPRE_Real *, comm_data->max_inflight[i], dmem_all_data->input.hypre_memory);
         for (int j = 0; j < comm_data->max_inflight[i]; j++){
            comm_data->data_inflight[i][j] = hypre_CTAlloc(HYPRE_Real, comm_data->len[i]+2, dmem_all_data->input.hypre_memory);
         }
      }
   }
   comm_data->data = hypre_CTAlloc(HYPRE_Real *, comm_data->procs.size(), dmem_all_data->input.hypre_memory);
   for (int i = 0; i < comm_data->procs.size(); i++){
      comm_data->data[i] = hypre_CTAlloc(HYPRE_Real, comm_data->len[i]+2, dmem_all_data->input.hypre_memory);
   }
}

// TODO: consolidate some redundant code
void CreateCommData_LocalRes(DMEM_AllData *dmem_all_data)
{
   int num_procs, my_id;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);   

   int my_grid = dmem_all_data->grid.my_grid;
   int finest_level = dmem_all_data->input.coarsest_mult_level;
   HYPRE_Int num_rows;

   hypre_ParAMGData *amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParAMGData *amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   hypre_ParCSRMatrix **A_array_gridk = hypre_ParAMGDataAArray(amg_data_gridk);
   hypre_ParCSRMatrix **A_array_fine = hypre_ParAMGDataAArray(amg_data_fine);

   int gridk_start = hypre_ParCSRMatrixFirstRowIndex(A_array_gridk[finest_level]);
   int gridk_end = hypre_ParCSRMatrixLastRowIndex(A_array_gridk[finest_level]);
   int my_gridk_part[2] = {gridk_start, gridk_end};
   int all_gridk_parts[2*num_procs];
   MPI_Allgather(my_gridk_part, 2, MPI_INT, all_gridk_parts, 2, MPI_INT, MPI_COMM_WORLD);

   int fine_start = hypre_ParCSRMatrixFirstRowIndex(A_array_fine[finest_level]);
   int fine_end = hypre_ParCSRMatrixLastRowIndex(A_array_fine[finest_level]);
   int my_fine_part[2] = {fine_start, fine_end};
   int all_fine_parts[2*num_procs];
   MPI_Allgather(my_fine_part, 2, MPI_INT, all_fine_parts, 2, MPI_INT, MPI_COMM_WORLD);

   dmem_all_data->grid.my_grid_procs_flags = (int *)calloc(num_procs, sizeof(int));
   my_grid = dmem_all_data->grid.my_grid;
   for (int i = 0; i < dmem_all_data->grid.num_procs_level[my_grid]; i++){
      int ip = dmem_all_data->grid.procs[my_grid][i];
      dmem_all_data->grid.my_grid_procs_flags[ip] = 1;
   }

/********************
 * FINE TO GRIDK
 ********************/

/* fine to gridk res inside send */
   dmem_all_data->comm.finestToGridk_Residual_insideSend.type = GRIDK_INSIDE_SEND;
   dmem_all_data->comm.finestToGridk_Residual_insideSend.tag = FINEST_TO_GRIDK_RESIDUAL_TAG;
   for (int p = 0; p < num_procs; p++){
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
         dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.push_back(p);
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_insideSend));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.size(); i++){
      int p = dmem_all_data->comm.finestToGridk_Residual_insideSend.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (p_gridk_start >= fine_start && p_gridk_start <= fine_end){
         dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] = p_gridk_start;
         dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] = fine_end;
      }
      else if (p_gridk_end <= fine_end && p_gridk_end >= fine_start){
         dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] = fine_start;
         dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] = p_gridk_end;
      }
      else if (p_gridk_end >= fine_end && p_gridk_start <= fine_start){
         dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] = fine_start;
         dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] = fine_end;
      }

      dmem_all_data->comm.finestToGridk_Residual_insideSend.len[i] =
         dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] - dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] + 1;
      dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] -= fine_start;
      dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] -= fine_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_insideSend));


/* fine to gridk res inside recv */
   dmem_all_data->comm.finestToGridk_Residual_insideRecv.type = GRIDK_INSIDE_RECV;
   dmem_all_data->comm.finestToGridk_Residual_insideRecv.tag = FINEST_TO_GRIDK_RESIDUAL_TAG;
   for (int p = 0; p < num_procs; p++){
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
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.procs.push_back(p);
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_insideRecv));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Residual_insideRecv.procs.size(); i++){
      int p = dmem_all_data->comm.finestToGridk_Residual_insideRecv.procs[i];
      int p_fine_start = all_fine_parts[2*p];
      int p_fine_end = all_fine_parts[2*p+1];
      if (gridk_start >= p_fine_start && gridk_start <= p_fine_end){
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] = gridk_start;
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] = p_fine_end;
      }
      else if (gridk_end <= p_fine_end && gridk_end >= p_fine_start){
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] = p_fine_start;
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] = gridk_end;
      }
      else if (gridk_start <= p_fine_start && gridk_end >= p_fine_end){
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] = p_fine_start;
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] = p_fine_end;
      }

      dmem_all_data->comm.finestToGridk_Residual_insideRecv.len[i] =
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] - dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] + 1;
      dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] -= gridk_start;
      dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] -= gridk_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_insideRecv));

/* fine to gridk correct inside send */
   dmem_all_data->comm.finestToGridk_Correct_insideSend.type = GRIDK_INSIDE_SEND;
   dmem_all_data->comm.finestToGridk_Correct_insideSend.tag = FINEST_TO_GRIDK_CORRECT_TAG;
   dmem_all_data->comm.finestToGridk_Correct_insideSend.procs = dmem_all_data->comm.finestToGridk_Residual_insideRecv.procs;
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideSend));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Correct_insideSend.procs.size(); i++){
      dmem_all_data->comm.finestToGridk_Correct_insideSend.start[i] = dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i];
      dmem_all_data->comm.finestToGridk_Correct_insideSend.end[i] = dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i];
      dmem_all_data->comm.finestToGridk_Correct_insideSend.len[i] = dmem_all_data->comm.finestToGridk_Residual_insideRecv.len[i];
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideSend));

/* fine to gridk correct inside recv */
   dmem_all_data->comm.finestToGridk_Correct_insideRecv.type = GRIDK_INSIDE_RECV;
   dmem_all_data->comm.finestToGridk_Correct_insideRecv.tag = FINEST_TO_GRIDK_CORRECT_TAG;
   dmem_all_data->comm.finestToGridk_Correct_insideRecv.procs = dmem_all_data->comm.finestToGridk_Residual_insideSend.procs;
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideRecv));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Correct_insideRecv.procs.size(); i++){
      dmem_all_data->comm.finestToGridk_Correct_insideRecv.start[i] = dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i];
      dmem_all_data->comm.finestToGridk_Correct_insideRecv.end[i] = dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i];
      dmem_all_data->comm.finestToGridk_Correct_insideRecv.len[i] = dmem_all_data->comm.finestToGridk_Residual_insideSend.len[i];
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideRecv));
   
/*********************
 * GRIDJ to GRIDK
 *********************/   
/* gridj to gridk correct inside send */
   dmem_all_data->comm.gridjToGridk_Correct_insideSend.type = GRIDK_INSIDE_SEND;
   dmem_all_data->comm.gridjToGridk_Correct_insideSend.tag = GRIDJ_TO_GRIDK_CORRECT_TAG;
   for (int p = 0; p < num_procs; p++){
      if (dmem_all_data->grid.my_grid_procs_flags[p] == 1){
         int p_gridk_start = all_gridk_parts[2*p];
         int p_gridk_end = all_gridk_parts[2*p+1];
         int flag = 0;
         if (p_gridk_end >= gridk_end && p_gridk_start <= gridk_start){
            flag = 1;
         }
         else if (p_gridk_end <= gridk_end && p_gridk_start >= gridk_start){
            flag = 1;
         }
         else if (p_gridk_start >= gridk_start && p_gridk_start <= gridk_end){
            flag = 1;
         }
         else if (p_gridk_end <= gridk_end && p_gridk_end >= gridk_start){
            flag = 1;
         }

         if (flag == 1){
            dmem_all_data->comm.gridjToGridk_Correct_insideSend.procs.push_back(p);
         }
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_insideSend));
   for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_insideSend.procs.size(); i++){
      int p = dmem_all_data->comm.gridjToGridk_Correct_insideSend.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (p_gridk_end >= gridk_end && p_gridk_start <= gridk_start){
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.start[i] = gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.end[i] = gridk_end;
      }
      else if (p_gridk_end <= gridk_end && p_gridk_start >= gridk_start){
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.start[i] = p_gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.end[i] = p_gridk_end;
      }
      else if (p_gridk_start >= gridk_start && p_gridk_start <= gridk_end){
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.start[i] = p_gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.end[i] = gridk_end;
      }
      else if (p_gridk_end <= gridk_end && p_gridk_end >= gridk_start){
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.start[i] = gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.end[i] = p_gridk_end;
      }

      dmem_all_data->comm.gridjToGridk_Correct_insideSend.len[i] =
         dmem_all_data->comm.gridjToGridk_Correct_insideSend.end[i] - dmem_all_data->comm.gridjToGridk_Correct_insideSend.start[i] + 1;
      dmem_all_data->comm.gridjToGridk_Correct_insideSend.start[i] -= gridk_start;
      dmem_all_data->comm.gridjToGridk_Correct_insideSend.end[i] -= gridk_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_insideSend));

/* gridj to gridk correct outside send */
   dmem_all_data->comm.gridjToGridk_Correct_outsideSend.type = GRIDK_OUTSIDE_SEND;
   dmem_all_data->comm.gridjToGridk_Correct_outsideSend.tag = GRIDJ_TO_GRIDK_CORRECT_TAG;
   for (int p = 0; p < num_procs; p++){
      if (dmem_all_data->grid.my_grid_procs_flags[p] == 0){
         int p_gridk_start = all_gridk_parts[2*p];
         int p_gridk_end = all_gridk_parts[2*p+1];
         int flag = 0;
         if (p_gridk_end >= gridk_end && p_gridk_start <= gridk_start){
            flag = 1;
         }
         else if (p_gridk_end <= gridk_end && p_gridk_start >= gridk_start){
            flag = 1;
         }
         else if (p_gridk_start >= gridk_start && p_gridk_start <= gridk_end){
            flag = 1;
         }
         else if (p_gridk_end <= gridk_end && p_gridk_end >= gridk_start){
            flag = 1;
         }

         if (flag == 1){
            dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.push_back(p);
         }
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend));
   for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs.size(); i++){
      int p = dmem_all_data->comm.gridjToGridk_Correct_outsideSend.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (p_gridk_end >= gridk_end && p_gridk_start <= gridk_start){
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.start[i] = gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.end[i] = gridk_end;
      }
      else if (p_gridk_end <= gridk_end && p_gridk_start >= gridk_start){
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.start[i] = p_gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.end[i] = p_gridk_end;
      }
      else if (p_gridk_start >= gridk_start && p_gridk_start <= gridk_end){
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.start[i] = p_gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.end[i] = gridk_end;
      }
      else if (p_gridk_end <= gridk_end && p_gridk_end >= gridk_start){
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.start[i] = gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.end[i] = p_gridk_end;
      }

      dmem_all_data->comm.gridjToGridk_Correct_outsideSend.len[i] =
         dmem_all_data->comm.gridjToGridk_Correct_outsideSend.end[i] - dmem_all_data->comm.gridjToGridk_Correct_outsideSend.start[i] + 1;
      dmem_all_data->comm.gridjToGridk_Correct_outsideSend.start[i] -= gridk_start;
      dmem_all_data->comm.gridjToGridk_Correct_outsideSend.end[i] -= gridk_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideSend));

/* gridj to gridk correct inside recv */
   dmem_all_data->comm.gridjToGridk_Correct_insideRecv.type = GRIDK_INSIDE_RECV;
   dmem_all_data->comm.gridjToGridk_Correct_insideRecv.tag = GRIDJ_TO_GRIDK_CORRECT_TAG;
   for (int p = 0; p < num_procs; p++){
      if (dmem_all_data->grid.my_grid_procs_flags[p] == 1){
         int flag = 0;
         int p_gridk_start = all_gridk_parts[2*p];
         int p_gridk_end = all_gridk_parts[2*p+1];
         if (gridk_start <= p_gridk_start && gridk_end >= p_gridk_end){
            flag = 1;
         }
         else if (gridk_start >= p_gridk_start && gridk_end <= p_gridk_end){
            flag = 1;
         }
         else if (gridk_start >= p_gridk_start && gridk_start <= p_gridk_end){
            flag = 1;
         }
         else if (gridk_end <= p_gridk_end && gridk_end >= p_gridk_start){
            flag = 1;
         }
   
         if (flag == 1){
            dmem_all_data->comm.gridjToGridk_Correct_insideRecv.procs.push_back(p);
         }
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_insideRecv));
   for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_insideRecv.procs.size(); i++){
      int p = dmem_all_data->comm.gridjToGridk_Correct_insideRecv.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (gridk_start <= p_gridk_start && gridk_end >= p_gridk_end){
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.start[i] = p_gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.end[i] = p_gridk_end;
      }
      else if (gridk_start >= p_gridk_start && gridk_end <= p_gridk_end){
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.start[i] = gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.end[i] = gridk_end;
      }
      else if (gridk_start >= p_gridk_start && gridk_start <= p_gridk_end){
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.start[i] = gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.end[i] = p_gridk_end;
      }
      else if (gridk_end <= p_gridk_end && gridk_end >= p_gridk_start){
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.start[i] = p_gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.end[i] = gridk_end;
      }

      dmem_all_data->comm.gridjToGridk_Correct_insideRecv.len[i] =
         dmem_all_data->comm.gridjToGridk_Correct_insideRecv.end[i] - dmem_all_data->comm.gridjToGridk_Correct_insideRecv.start[i] + 1;
      dmem_all_data->comm.gridjToGridk_Correct_insideRecv.start[i] -= gridk_start;
      dmem_all_data->comm.gridjToGridk_Correct_insideRecv.end[i] -= gridk_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_insideRecv));

/* gridj to gridk correct outside recv */
   dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.type = GRIDK_OUTSIDE_RECV;
   dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.tag = GRIDJ_TO_GRIDK_CORRECT_TAG;
   for (int p = 0; p < num_procs; p++){
      if (dmem_all_data->grid.my_grid_procs_flags[p] == 0){
         int flag = 0;
         int p_gridk_start = all_gridk_parts[2*p];
         int p_gridk_end = all_gridk_parts[2*p+1];
         if (gridk_start <= p_gridk_start && gridk_end >= p_gridk_end){
            flag = 1;
         }
         else if (gridk_start >= p_gridk_start && gridk_end <= p_gridk_end){
            flag = 1;
         }
         else if (gridk_start >= p_gridk_start && gridk_start <= p_gridk_end){
            flag = 1;
         }
         else if (gridk_end <= p_gridk_end && gridk_end >= p_gridk_start){
            flag = 1;
         }
   
         if (flag == 1){
            dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.procs.push_back(p);
         }
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv));
   for (int i = 0; i < dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.procs.size(); i++){
      int p = dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (gridk_start <= p_gridk_start && gridk_end >= p_gridk_end){
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.start[i] = p_gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.end[i] = p_gridk_end;
      }
      else if (gridk_start >= p_gridk_start && gridk_end <= p_gridk_end){
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.start[i] = gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.end[i] = gridk_end;
      }
      else if (gridk_start >= p_gridk_start && gridk_start <= p_gridk_end){
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.start[i] = gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.end[i] = p_gridk_end;
      }
      else if (gridk_end <= p_gridk_end && gridk_end >= p_gridk_start){
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.start[i] = p_gridk_start;
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.end[i] = gridk_end;
      }

      dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.len[i] =
         dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.end[i] - dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.start[i] + 1;
      dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.start[i] -= gridk_start;
      dmem_all_data->comm.gridjToGridk_Correct_outsideRecv.end[i] -= gridk_start;
   }
   AllocCommData(dmem_all_data,  &(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv));

   /************************************
    * FINE FOR ASYNC SMOOTHING
    ************************************/
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(dmem_all_data->matrix.A_gridk);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   num_rows = hypre_ParCSRMatrixNumRows(dmem_all_data->matrix.A_gridk);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(dmem_all_data->matrix.A_gridk);
   HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int *A_offd_j = hypre_CSRMatrixJ(A_offd);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(dmem_all_data->matrix.A_gridk);
   HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);

   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   dmem_all_data->vector_gridk.x_ghost = hypre_SeqVectorCreate(num_cols_offd);
   hypre_SeqVectorInitialize(dmem_all_data->vector_gridk.x_ghost);
   dmem_all_data->vector_gridk.x_ghost_prev = hypre_SeqVectorCreate(num_cols_offd);
   hypre_SeqVectorInitialize(dmem_all_data->vector_gridk.x_ghost_prev);
  // dmem_all_data->vector_gridk.b_ghost = hypre_SeqVectorCreate(num_cols_offd);
  // dmem_all_data->vector_gridk.a_diag_ghost = hypre_SeqVectorCreate(num_cols_offd);
  // dmem_all_data->vector_gridk.a_diag = hypre_SeqVectorCreate(num_rows);
  // hypre_SeqVectorInitialize(dmem_all_data->vector_gridk.b_ghost);
  // hypre_SeqVectorInitialize(dmem_all_data->vector_gridk.a_diag_ghost);
  // hypre_SeqVectorInitialize(dmem_all_data->vector_gridk.a_diag);
  // HYPRE_Real *a_diag_data = hypre_VectorData(dmem_all_data->vector_gridk.a_diag);
   int j;
/* fine inside send */
   if (dmem_all_data->input.async_flag == 0){
      dmem_all_data->comm.finestIntra_outsideSend.type = FINE_INTRA_INSIDE_SEND;
   }
   else {
      dmem_all_data->comm.finestIntra_outsideSend.type = FINE_INTRA_OUTSIDE_SEND;
   }
   dmem_all_data->comm.finestIntra_outsideSend.tag = FINE_INTRA_TAG;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      dmem_all_data->comm.finestIntra_outsideSend.procs.push_back(ip);
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend));
   j = 0;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      dmem_all_data->comm.finestIntra_outsideSend.start[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      dmem_all_data->comm.finestIntra_outsideSend.end[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
      dmem_all_data->comm.finestIntra_outsideSend.len[j] =
         dmem_all_data->comm.finestIntra_outsideSend.end[j] - dmem_all_data->comm.finestIntra_outsideSend.start[j];
      j++;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend));

/* fine outside recv */
   if (dmem_all_data->input.async_flag == 0){
      dmem_all_data->comm.finestIntra_outsideRecv.type = FINE_INTRA_INSIDE_RECV;
   }
   else {
      dmem_all_data->comm.finestIntra_outsideRecv.type = FINE_INTRA_OUTSIDE_RECV;
   }
   dmem_all_data->comm.finestIntra_outsideRecv.tag = FINE_INTRA_TAG;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      dmem_all_data->comm.finestIntra_outsideRecv.procs.push_back(ip);
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));
   j = 0;
   dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_data.resize(num_recvs);
   dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_j.resize(num_recvs);
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
//#if defined(HYPRE_USING_CUDA)
//#else
      dmem_all_data->comm.finestIntra_outsideRecv.start[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      dmem_all_data->comm.finestIntra_outsideRecv.end[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i+1);
      dmem_all_data->comm.finestIntra_outsideRecv.len[j] =
         dmem_all_data->comm.finestIntra_outsideRecv.end[j] - dmem_all_data->comm.finestIntra_outsideRecv.start[j];
//#endif

      dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_data[j].resize(num_rows);
      dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_j[j].resize(num_rows);
      for (int k = 0; k < num_rows; k++){
        // a_diag_data[k] = A_diag_data[A_diag_i[k]];
         for (int jj = A_offd_i[k]; jj < A_offd_i[k+1]; jj++){
            int ii = A_offd_j[jj];
            if (ii >= dmem_all_data->comm.finestIntra_outsideRecv.start[j] &&
                ii < dmem_all_data->comm.finestIntra_outsideRecv.end[j]){
               dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_data[j][k].push_back(A_offd_data[jj]);
               dmem_all_data->comm.finestIntra_outsideRecv.a_ghost_j[j][k].push_back(ii);
            }
         }
      }
      j++;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));
}

void CreateCommData_GlobalRes(DMEM_AllData *dmem_all_data)
{
   int num_procs, my_id;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);   

   int my_grid = dmem_all_data->grid.my_grid;

   int finest_level = dmem_all_data->input.coarsest_mult_level;

   hypre_ParAMGData *amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParAMGData *amg_data_gridk = (hypre_ParAMGData *)dmem_all_data->hypre.solver_gridk;
   hypre_ParCSRMatrix **A_array_gridk = hypre_ParAMGDataAArray(amg_data_gridk);
   hypre_ParCSRMatrix **A_array_fine = hypre_ParAMGDataAArray(amg_data_fine);

   int gridk_start = hypre_ParCSRMatrixFirstRowIndex(A_array_gridk[finest_level]);
   int gridk_end = hypre_ParCSRMatrixLastRowIndex(A_array_gridk[finest_level]);
   int my_gridk_part[2] = {gridk_start, gridk_end};
   int all_gridk_parts[2*num_procs];
   MPI_Allgather(my_gridk_part, 2, MPI_INT, all_gridk_parts, 2, MPI_INT, MPI_COMM_WORLD);

   int fine_start = hypre_ParCSRMatrixFirstRowIndex(A_array_fine[finest_level]);
   int fine_end = hypre_ParCSRMatrixLastRowIndex(A_array_fine[finest_level]);
   int my_fine_part[2] = {fine_start, fine_end};
   int all_fine_parts[2*num_procs];
   MPI_Allgather(my_fine_part, 2, MPI_INT, all_fine_parts, 2, MPI_INT, MPI_COMM_WORLD);

   dmem_all_data->grid.my_grid_procs_flags = (int *)calloc(num_procs, sizeof(int));
   my_grid = dmem_all_data->grid.my_grid;
   for (int i = 0; i < dmem_all_data->grid.num_procs_level[my_grid]; i++){
      int ip = dmem_all_data->grid.procs[my_grid][i];
      dmem_all_data->grid.my_grid_procs_flags[ip] = 1;
   }
   MPI_Barrier(MPI_COMM_WORLD);
  // MPI_Barrier(MPI_COMM_WORLD);

/**************************************
 * gridk res
 **************************************/

/* gridk res inside send */
   dmem_all_data->comm.finestToGridk_Residual_insideSend.type = GRIDK_INSIDE_SEND;
   dmem_all_data->comm.finestToGridk_Residual_insideSend.tag = FINEST_TO_GRIDK_RESIDUAL_TAG;
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
            dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.push_back(p);
         }
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_insideSend));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Residual_insideSend.procs.size(); i++){
      int p = dmem_all_data->comm.finestToGridk_Residual_insideSend.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (p_gridk_start >= fine_start && p_gridk_start <= fine_end){
         dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] = p_gridk_start;
         dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] = fine_end;
      }
      else if (p_gridk_end <= fine_end && p_gridk_end >= fine_start){
         dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] = fine_start;
         dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] = p_gridk_end;
      }
      else if (p_gridk_end >= fine_end && p_gridk_start <= fine_start){
         dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] = fine_start;
         dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] = fine_end;
      }

      dmem_all_data->comm.finestToGridk_Residual_insideSend.len[i] =
         dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] - dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] + 1;
      dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i] -= fine_start;
      dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i] -= fine_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_insideSend));

/* gridk res outside send */
   dmem_all_data->comm.finestToGridk_Residual_outsideSend.type = GRIDK_OUTSIDE_SEND;
   dmem_all_data->comm.finestToGridk_Residual_outsideSend.tag = FINEST_TO_GRIDK_RESIDUAL_TAG;
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
            dmem_all_data->comm.finestToGridk_Residual_outsideSend.procs.push_back(p);
         }
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideSend));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Residual_outsideSend.procs.size(); i++){
      int p = dmem_all_data->comm.finestToGridk_Residual_outsideSend.procs[i];
      int p_gridk_start = all_gridk_parts[2*p];
      int p_gridk_end = all_gridk_parts[2*p+1];
      if (p_gridk_start >= fine_start && p_gridk_start <= fine_end){
         dmem_all_data->comm.finestToGridk_Residual_outsideSend.start[i] = p_gridk_start;
         dmem_all_data->comm.finestToGridk_Residual_outsideSend.end[i] = fine_end;
      }
      else if (p_gridk_end <= fine_end && p_gridk_end >= fine_start){
         dmem_all_data->comm.finestToGridk_Residual_outsideSend.start[i] = fine_start;
         dmem_all_data->comm.finestToGridk_Residual_outsideSend.end[i] = p_gridk_end;
      }
      else if (p_gridk_end >= fine_end && p_gridk_start <= fine_start){
         dmem_all_data->comm.finestToGridk_Residual_outsideSend.start[i] = fine_start;
         dmem_all_data->comm.finestToGridk_Residual_outsideSend.end[i] = fine_end;
      }

      dmem_all_data->comm.finestToGridk_Residual_outsideSend.len[i] =
         dmem_all_data->comm.finestToGridk_Residual_outsideSend.end[i] - dmem_all_data->comm.finestToGridk_Residual_outsideSend.start[i] + 1;
      dmem_all_data->comm.finestToGridk_Residual_outsideSend.start[i] -= fine_start;
      dmem_all_data->comm.finestToGridk_Residual_outsideSend.end[i] -= fine_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideSend));

/* gridk res inside recv */
   dmem_all_data->comm.finestToGridk_Residual_insideRecv.type = GRIDK_INSIDE_RECV;
   dmem_all_data->comm.finestToGridk_Residual_insideRecv.tag = FINEST_TO_GRIDK_RESIDUAL_TAG;
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
            dmem_all_data->comm.finestToGridk_Residual_insideRecv.procs.push_back(p);
         }
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_insideRecv));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Residual_insideRecv.procs.size(); i++){
      int p = dmem_all_data->comm.finestToGridk_Residual_insideRecv.procs[i];
      int p_fine_start = all_fine_parts[2*p];
      int p_fine_end = all_fine_parts[2*p+1];
      if (gridk_start >= p_fine_start && gridk_start <= p_fine_end){
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] = gridk_start;
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] = p_fine_end;
      }
      else if (gridk_end <= p_fine_end && gridk_end >= p_fine_start){
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] = p_fine_start;
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] = gridk_end;
      }
      else if (gridk_start <= p_fine_start && gridk_end >= p_fine_end){
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] = p_fine_start;
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] = p_fine_end;
      }

      dmem_all_data->comm.finestToGridk_Residual_insideRecv.len[i] =
         dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] - dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] + 1;
      dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i] -= gridk_start;
      dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i] -= gridk_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_insideRecv));


/* gridk res outside recv */
   dmem_all_data->comm.finestToGridk_Residual_outsideRecv.type = GRIDK_OUTSIDE_RECV;
   dmem_all_data->comm.finestToGridk_Residual_outsideRecv.tag = FINEST_TO_GRIDK_RESIDUAL_TAG;
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
            dmem_all_data->comm.finestToGridk_Residual_outsideRecv.procs.push_back(p);
         }
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Residual_outsideRecv.procs.size(); i++){
      int p = dmem_all_data->comm.finestToGridk_Residual_outsideRecv.procs[i];
      int p_fine_start = all_fine_parts[2*p];
      int p_fine_end = all_fine_parts[2*p+1];
      if (gridk_start >= p_fine_start && gridk_start <= p_fine_end){
         dmem_all_data->comm.finestToGridk_Residual_outsideRecv.start[i] = gridk_start;
         dmem_all_data->comm.finestToGridk_Residual_outsideRecv.end[i] = p_fine_end;
      }
      else if (gridk_end <= p_fine_end && gridk_end >= p_fine_start){
         dmem_all_data->comm.finestToGridk_Residual_outsideRecv.start[i] = p_fine_start;
         dmem_all_data->comm.finestToGridk_Residual_outsideRecv.end[i] = gridk_end;
      }
      else if (gridk_start <= p_fine_start && gridk_end >= p_fine_end){
         dmem_all_data->comm.finestToGridk_Residual_outsideRecv.start[i] = p_fine_start;
         dmem_all_data->comm.finestToGridk_Residual_outsideRecv.end[i] = p_fine_end;
      }

      dmem_all_data->comm.finestToGridk_Residual_outsideRecv.len[i] =
         dmem_all_data->comm.finestToGridk_Residual_outsideRecv.end[i] - dmem_all_data->comm.finestToGridk_Residual_outsideRecv.start[i] + 1;
      dmem_all_data->comm.finestToGridk_Residual_outsideRecv.start[i] -= gridk_start;
      dmem_all_data->comm.finestToGridk_Residual_outsideRecv.end[i] -= gridk_start;
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Residual_outsideRecv));

/**************************************
 * gridk correct
 **************************************/

/* gridk correct inside send */
   dmem_all_data->comm.finestToGridk_Correct_insideSend.type = GRIDK_INSIDE_SEND;
   dmem_all_data->comm.finestToGridk_Correct_insideSend.tag = FINEST_TO_GRIDK_CORRECT_TAG;
   dmem_all_data->comm.finestToGridk_Correct_insideSend.procs = dmem_all_data->comm.finestToGridk_Residual_insideRecv.procs;
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideSend));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Correct_insideSend.procs.size(); i++){
      dmem_all_data->comm.finestToGridk_Correct_insideSend.start[i] = dmem_all_data->comm.finestToGridk_Residual_insideRecv.start[i];
      dmem_all_data->comm.finestToGridk_Correct_insideSend.end[i] = dmem_all_data->comm.finestToGridk_Residual_insideRecv.end[i];
      dmem_all_data->comm.finestToGridk_Correct_insideSend.len[i] = dmem_all_data->comm.finestToGridk_Residual_insideRecv.len[i];
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideSend));

/* gridk correct outside send */
   dmem_all_data->comm.finestToGridk_Correct_outsideSend.type = GRIDK_OUTSIDE_SEND;
   dmem_all_data->comm.finestToGridk_Correct_outsideSend.tag = FINEST_TO_GRIDK_CORRECT_TAG;
   dmem_all_data->comm.finestToGridk_Correct_outsideSend.procs = dmem_all_data->comm.finestToGridk_Residual_outsideRecv.procs;
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideSend));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Correct_outsideSend.procs.size(); i++){
      dmem_all_data->comm.finestToGridk_Correct_outsideSend.start[i] = dmem_all_data->comm.finestToGridk_Residual_outsideRecv.start[i];
      dmem_all_data->comm.finestToGridk_Correct_outsideSend.end[i] = dmem_all_data->comm.finestToGridk_Residual_outsideRecv.end[i];
      dmem_all_data->comm.finestToGridk_Correct_outsideSend.len[i] = dmem_all_data->comm.finestToGridk_Residual_outsideRecv.len[i];
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideSend));

/* gridk correct inside recv */
   dmem_all_data->comm.finestToGridk_Correct_insideRecv.type = GRIDK_INSIDE_RECV;
   dmem_all_data->comm.finestToGridk_Correct_insideRecv.tag = FINEST_TO_GRIDK_CORRECT_TAG;
   dmem_all_data->comm.finestToGridk_Correct_insideRecv.procs = dmem_all_data->comm.finestToGridk_Residual_insideSend.procs;
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideRecv));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Correct_insideRecv.procs.size(); i++){
      dmem_all_data->comm.finestToGridk_Correct_insideRecv.start[i] = dmem_all_data->comm.finestToGridk_Residual_insideSend.start[i];
      dmem_all_data->comm.finestToGridk_Correct_insideRecv.end[i] = dmem_all_data->comm.finestToGridk_Residual_insideSend.end[i];
      dmem_all_data->comm.finestToGridk_Correct_insideRecv.len[i] = dmem_all_data->comm.finestToGridk_Residual_insideSend.len[i];
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideRecv));

/* gridk correct outside recv */
   dmem_all_data->comm.finestToGridk_Correct_outsideRecv.type = GRIDK_OUTSIDE_RECV;
   dmem_all_data->comm.finestToGridk_Correct_outsideRecv.tag = FINEST_TO_GRIDK_CORRECT_TAG;
   dmem_all_data->comm.finestToGridk_Correct_outsideRecv.procs = dmem_all_data->comm.finestToGridk_Residual_outsideSend.procs;
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_outsideRecv));
   for (int i = 0; i < dmem_all_data->comm.finestToGridk_Correct_outsideRecv.procs.size(); i++){
      dmem_all_data->comm.finestToGridk_Correct_outsideRecv.start[i] = dmem_all_data->comm.finestToGridk_Residual_outsideSend.start[i];
      dmem_all_data->comm.finestToGridk_Correct_outsideRecv.end[i] = dmem_all_data->comm.finestToGridk_Residual_outsideSend.end[i];
      dmem_all_data->comm.finestToGridk_Correct_outsideRecv.len[i] = dmem_all_data->comm.finestToGridk_Residual_outsideSend.len[i];
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestToGridk_Correct_insideRecv));

/************************************
 * FINE
 ************************************/
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(dmem_all_data->matrix.A_fine);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   dmem_all_data->comm.fine_send_data = hypre_CTAlloc(HYPRE_Real, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends), dmem_all_data->input.hypre_memory);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(dmem_all_data->matrix.A_fine);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
   dmem_all_data->comm.fine_recv_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, dmem_all_data->input.hypre_memory);
   dmem_all_data->vector_fine.x_ghost = hypre_SeqVectorCreate(num_cols_offd);
   hypre_SeqVectorInitialize(dmem_all_data->vector_fine.x_ghost);
   int j;

/* fine inside send */
   dmem_all_data->comm.finestIntra_insideSend.type = FINE_INTRA_INSIDE_SEND;
   dmem_all_data->comm.finestIntra_insideSend.tag = FINE_INTRA_TAG;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 1){
         dmem_all_data->comm.finestIntra_insideSend.procs.push_back(ip);
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestIntra_insideSend));
   j = 0;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 1){
         dmem_all_data->comm.finestIntra_insideSend.start[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         dmem_all_data->comm.finestIntra_insideSend.end[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
         dmem_all_data->comm.finestIntra_insideSend.len[j] =
            dmem_all_data->comm.finestIntra_insideSend.end[j] - dmem_all_data->comm.finestIntra_insideSend.start[j];
         j++;
      }
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestIntra_insideSend));

/* fine outside send */
   dmem_all_data->comm.finestIntra_outsideSend.type = FINE_INTRA_OUTSIDE_SEND;
   dmem_all_data->comm.finestIntra_outsideSend.tag = FINE_INTRA_TAG;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 0){
         dmem_all_data->comm.finestIntra_outsideSend.procs.push_back(ip);
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend));
   j = 0;
   for (int i = 0; i < num_sends; i++){
      int ip = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 0){
         dmem_all_data->comm.finestIntra_outsideSend.start[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         dmem_all_data->comm.finestIntra_outsideSend.end[j] = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
         dmem_all_data->comm.finestIntra_outsideSend.len[j] =
            dmem_all_data->comm.finestIntra_outsideSend.end[j] - dmem_all_data->comm.finestIntra_outsideSend.start[j];
         j++;
      }
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideSend));

/* fine inside recv */
   dmem_all_data->comm.finestIntra_insideRecv.type = FINE_INTRA_INSIDE_RECV;
   dmem_all_data->comm.finestIntra_insideRecv.tag = FINE_INTRA_TAG;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 1){
         dmem_all_data->comm.finestIntra_insideRecv.procs.push_back(ip);
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestIntra_insideRecv));
   j = 0;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 1){
         dmem_all_data->comm.finestIntra_insideRecv.start[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
         dmem_all_data->comm.finestIntra_insideRecv.end[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i+1);
         dmem_all_data->comm.finestIntra_insideRecv.len[j] =
            dmem_all_data->comm.finestIntra_insideRecv.end[j] - dmem_all_data->comm.finestIntra_insideRecv.start[j];
         j++;
      }
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestIntra_insideRecv));

/* fine outside recv */
   dmem_all_data->comm.finestIntra_outsideRecv.type = FINE_INTRA_OUTSIDE_RECV;
   dmem_all_data->comm.finestIntra_outsideRecv.tag = FINE_INTRA_TAG;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 0){
         dmem_all_data->comm.finestIntra_outsideRecv.procs.push_back(ip);
      }
   }
   AllocCommVars(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));
   j = 0;
   for (int i = 0; i < num_recvs; i++){
      int ip = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      if (dmem_all_data->grid.my_grid_procs_flags[ip] == 0){
         dmem_all_data->comm.finestIntra_outsideRecv.start[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
         dmem_all_data->comm.finestIntra_outsideRecv.end[j] = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i+1);
         dmem_all_data->comm.finestIntra_outsideRecv.len[j] =
            dmem_all_data->comm.finestIntra_outsideRecv.end[j] - dmem_all_data->comm.finestIntra_outsideRecv.start[j];
         j++;
      }
   }
   AllocCommData(dmem_all_data, &(dmem_all_data->comm.finestIntra_outsideRecv));

  // if (my_id == 0) printf("inside send:\n");
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("\t%d, %d:", p, my_grid);
  //       for (int i = 0; i < dmem_all_data->comm.finestIntra_insideSend.procs.size(); i++){
  //          printf(" %d", dmem_all_data->comm.finestIntra_insideSend.procs[i]);
  //       }
  //       printf("\n");
  //    }
  //    sleep(1);
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
  // if (my_id == 0) printf("inside recv:\n");
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("\t%d, %d:", p, my_grid);
  //       for (int i = 0; i < dmem_all_data->comm.finestIntra_insideRecv.procs.size(); i++){
  //          printf(" %d", dmem_all_data->comm.finestIntra_insideRecv.procs[i]);
  //       }
  //       printf("\n");
  //    }
  //    sleep(1);
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }

  // if (my_id == 0) printf("\n\noutside send:\n");
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("\t%d, %d:", p, my_grid);
  //       for (int i = 0; i < dmem_all_data->comm.finestIntra_outsideSend.procs.size(); i++){
  //          printf(" %d", dmem_all_data->comm.finestIntra_outsideSend.procs[i]);
  //       }
  //       printf("\n");
  //    }
  //    sleep(1);
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
  // if (my_id == 0) printf("outside recv:\n");
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("\t%d, %d:", p, my_grid);
  //       for (int i = 0; i < dmem_all_data->comm.finestIntra_outsideRecv.procs.size(); i++){
  //          printf(" %d", dmem_all_data->comm.finestIntra_outsideRecv.procs[i]);
  //       }
  //       printf("\n");
  //    }
  //    sleep(1);
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
}

void DistributeMatrix(DMEM_AllData *dmem_all_data,
                      hypre_ParCSRMatrix *A,
                      hypre_ParCSRMatrix **B)
{
   HYPRE_Int num_procs, my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   MPI_Comm my_comm = dmem_all_data->grid.my_comm;

   HYPRE_Int loc_num_procs, loc_my_id;
   MPI_Comm_rank(my_comm, &loc_my_id);
   MPI_Comm_size(my_comm, &loc_num_procs);

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;
   HYPRE_Int num_levels = dmem_all_data->grid.num_levels;
   
   HYPRE_Int num_procs_level, rest, **ps, **pe;
   ps = (HYPRE_Int **)malloc(num_levels * sizeof(HYPRE_Int *));
   pe = (HYPRE_Int **)malloc(num_levels * sizeof(HYPRE_Int *));
   //printf("%d, %d, %d\n", num_procs, num_my_procs, rest);

   int finest_level;
   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      finest_level = dmem_all_data->input.coarsest_mult_level + 1;
   }
   else{
      finest_level = dmem_all_data->input.coarsest_mult_level;
   }
   for (int level = finest_level; level < num_levels; level++){
      ps[level] = (HYPRE_Int *)calloc(dmem_all_data->grid.num_procs_level[level], sizeof(HYPRE_Int));
      pe[level] = (HYPRE_Int *)calloc(dmem_all_data->grid.num_procs_level[level], sizeof(HYPRE_Int));

      num_procs_level = num_procs/dmem_all_data->grid.num_procs_level[level];
      rest = num_procs - num_procs_level*dmem_all_data->grid.num_procs_level[level];
      for (int p = 0; p < dmem_all_data->grid.num_procs_level[level]; p++){
         if (p < rest){
            ps[level][p] = p*num_procs_level + p;
            pe[level][p] = (p + 1)*num_procs_level + p + 1;
         }
         else {
            ps[level][p] = p*num_procs_level + rest;
            pe[level][p] = (p + 1)*num_procs_level + rest;
         }
      }
   }

  // printf("(%d,%d,%d): %d, %d\n", loc_my_id, my_grid, num_my_procs, ps, pe)

   double *recvbuf_v, *sendbuf_v;
   int *recvbuf_i, *sendbuf_i;
   int *recvbuf_j, *sendbuf_j;
   int *recvbuf, *sendbuf;
   int *rdispls, *recvcounts;
   int *sdispls, *sendcounts;
   int sendcount, recvcount;

   recvcounts = (int *)calloc(num_procs, sizeof(int));
   rdispls = (int *)calloc(num_procs, sizeof(int));
   sendcounts = (int *)calloc(num_procs, sizeof(int));
   sdispls = (int *)calloc(num_procs, sizeof(int));

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Int *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Real *diag_data = hypre_CSRMatrixData(diag);

   HYPRE_Int *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Real *offd_data = hypre_CSRMatrixData(offd);

   /* indicate to other processes that I need their diag and offd */
   recvbuf = (int *)calloc(num_procs, sizeof(int));
   sendbuf = (int *)calloc(num_procs, sizeof(int));

   HYPRE_Int *recv_flags = (HYPRE_Int *)calloc(num_procs, sizeof(HYPRE_Int));
   for (HYPRE_Int p = 0; p < num_procs; p++){
      if (p >= ps[my_grid][loc_my_id] && p < pe[my_grid][loc_my_id]){
         sendbuf[p] = 1;
         recv_flags[p] = 1;
      }
   }
  
   MPI_Alltoall(sendbuf,
                1,
                MPI_INT,
                recvbuf,
                1,
                MPI_INT,
                MPI_COMM_WORLD);

   /* send nnz */
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(diag) + hypre_CSRMatrixNumNonzeros(offd);
   int *send_flags = (int *)calloc(num_procs, sizeof(int));
   sendcount = 0;
   for (int p = 0; p < num_procs; p++){
      sendbuf[p] = 0;
      if (recvbuf[p] == 1){
         send_flags[p] = 1;
         sendbuf[p] = nnz;
      }
      /* we need sdispls and sendcount later when we send I,J,V */
      if (p > 0){
         sdispls[p] = sdispls[p-1] + sendbuf[p-1];
      }
      sendcounts[p] = sendbuf[p];
      sendcount += sendbuf[p];
   }

   MPI_Alltoall(sendbuf,
                1,
                MPI_INT,
                recvbuf,
                1,
                MPI_INT,
                MPI_COMM_WORLD);


  // if (my_id == 1)
  // for (HYPRE_Int p = 0; p < num_procs; p++) printf("%d\n", recvbuf[p]);

 
   /* send I,J,V */
   recvcount = 0;
   for (HYPRE_Int p = 0; p < num_procs; p++){
      if (p > 0){
         rdispls[p] = rdispls[p-1] + recvbuf[p-1];
      }
      recvcounts[p] = recvbuf[p];
      recvcount += recvbuf[p];
   }

   free(sendbuf);
   free(recvbuf);

   recvbuf_i = (int *)calloc(recvcount, sizeof(int));
   recvbuf_j = (int *)calloc(recvcount, sizeof(int));
   recvbuf_v = (double *)calloc(recvcount, sizeof(double));

   sendbuf_j = (int *)calloc(sendcount, sizeof(int));
   sendbuf_i = (int *)calloc(sendcount, sizeof(int));
   sendbuf_v = (double *)calloc(sendcount, sizeof(double));

  // HYPRE_Int *I = (HYPRE_Int *)calloc(recvcount, sizeof(HYPRE_Int));
  // HYPRE_Int *J = (HYPRE_Int *)calloc(recvcount, sizeof(HYPRE_Int));
  // HYPRE_Real *V = (HYPRE_Real *)calloc(recvcount, sizeof(HYPRE_Real));

 
   HYPRE_Int k = 0;
   for (HYPRE_Int p = 0; p < num_procs; p++){
      if (send_flags[p] == 1){

         HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
         HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
         HYPRE_Int num_rows = hypre_CSRMatrixNumRows(diag);
         HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

         for (HYPRE_Int i = 0; i < num_rows; i++){
            for (HYPRE_Int jj = diag_i[i]; jj < diag_i[i+1]; jj++){
               HYPRE_Int ii = diag_j[jj];

               sendbuf_i[k] = first_row_index+i;
               sendbuf_j[k] = first_col_diag+ii;
               sendbuf_v[k] = diag_data[jj];

               k++;
            }
         }

         for (HYPRE_Int i = 0; i < num_rows; i++){
            for (HYPRE_Int jj = offd_i[i]; jj < offd_i[i+1]; jj++){
               HYPRE_Int ii = offd_j[jj];

               sendbuf_i[k] = first_row_index+i;
               sendbuf_j[k] = col_map_offd[ii];
               sendbuf_v[k] = offd_data[jj];

               k++;
            }
         }
      }
   }
 
   MPI_Alltoallv(sendbuf_i,
                 sendcounts,
                 sdispls,
                 MPI_INT,
                 recvbuf_i,
                 recvcounts,
                 rdispls,
                 MPI_INT,
                 MPI_COMM_WORLD);

   MPI_Alltoallv(sendbuf_j,
                 sendcounts,
                 sdispls,
                 MPI_INT,
                 recvbuf_j,
                 recvcounts,
                 rdispls,
                 MPI_INT,
                 MPI_COMM_WORLD);

   MPI_Alltoallv(sendbuf_v,
                 sendcounts,
                 sdispls,
                 MPI_DOUBLE,
                 recvbuf_v,
                 recvcounts,
                 rdispls,
                 MPI_DOUBLE,
                 MPI_COMM_WORLD);

   
   HYPRE_IJMatrix ij_matrix;
   int ilower = MinInt(recvbuf_i, recvcount);
   int iupper = MaxInt(recvbuf_i, recvcount);
   int jlower = MinInt(recvbuf_j, recvcount);
   int jupper = MaxInt(recvbuf_j, recvcount);

   recvbuf = (int *)calloc(4*num_procs, sizeof(int));
   sendbuf = (int *)calloc(4*num_procs, sizeof(int));

   for (int p = 0; p < num_procs; p++){
      if (send_flags[p] == 1){
         int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
         int last_row_index = hypre_ParCSRMatrixLastRowIndex(A);
         int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
         int last_col_diag = hypre_ParCSRMatrixLastColDiag(A);

         sendbuf[4*p] =   first_row_index;
         sendbuf[4*p+1] = last_row_index;
         sendbuf[4*p+2] = first_col_diag;
         sendbuf[4*p+3] = last_col_diag;
      }
   }
   
  // printf("%d: interp %d %d\n", my_id, iupper, jupper);

   MPI_Alltoall(sendbuf,
                4,
                MPI_INT,
                recvbuf,
                4,
                MPI_INT,
                MPI_COMM_WORLD);

   ilower = jlower = INT_MAX;//max(hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixGlobalNumCols(A))+1;
   iupper = jupper = INT_MIN;
   for (int p = 0; p < num_procs; p++){
      if (recv_flags[p] == 1){
         if (ilower > recvbuf[4*p]){
            ilower = recvbuf[4*p];
         }
         if (iupper < recvbuf[4*p+1]){
            iupper = recvbuf[4*p+1];
         }
         if (jlower > recvbuf[4*p+2]){
            jlower = recvbuf[4*p+2];
         }
         if (jupper < recvbuf[4*p+3]){
            jupper = recvbuf[4*p+3];
         }
      }
   }
   
   free(sendbuf);
   free(recvbuf);
   

   HYPRE_IJMatrixCreate(my_comm, ilower, iupper, jlower, jupper, &ij_matrix);
   HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(ij_matrix);


  // printf("%d, %d\n", ilower, iupper);

   HYPRE_Int num_rows = iupper-ilower+1;
   vector<vector<int>> col_vec(num_rows, vector<int>(0));
   vector<vector<double>> val_vec(num_rows, vector<double>(0));
   for (HYPRE_Int k = 0; k < recvcount; k++){
      HYPRE_Int row_ind = recvbuf_i[k]-ilower;
      col_vec[row_ind].push_back(recvbuf_j[k]);
      val_vec[row_ind].push_back(recvbuf_v[k]);
   }
   for (HYPRE_Int i = 0; i < num_rows; i++){
      int ncols = col_vec[i].size(); 
      int *cols = (int *)calloc(ncols, sizeof(int));
      double *values = (double *)calloc(ncols, sizeof(double));
      for (HYPRE_Int j = 0; j < ncols; j++){
         cols[j] = col_vec[i].back();
         values[j] = val_vec[i].back();
         col_vec[i].pop_back();
         val_vec[i].pop_back();
      }
      int I = ilower+i;
      HYPRE_IJMatrixSetValues(ij_matrix, 1, &ncols, &I, cols, values);
      free(cols);
      free(values);
   }

   HYPRE_IJMatrixAssemble(ij_matrix);
   HYPRE_IJMatrixGetObject(ij_matrix, (void**)B);

  // char buffer[100];
  // sprintf(buffer, "A_async_%d.txt", my_grid);
  // DMEM_PrintParCSRMatrix(*B, buffer);

  // if (my_id == 1){
  //    for (HYPRE_Int k = 0; k < recvcount; k++){
  //       printf("%d, %d: %d %d %e\n", my_id, k, recvbuf_i[k], recvbuf_j[k], recvbuf_v[k]);
  //    }
  //    printf("%d, %d\n", ilower, iupper);

  //   // for (HYPRE_Int k = 0; k < sendcount; k++){
  //   //    printf("%d %d %e\n", sendbuf_i[k], sendbuf_j[k], sendbuf_v[k]);
  //   // }
  //    
  //   // for (HYPRE_Int p = 0; p < num_procs; p++){
  //   //     printf("%d %d\n", sdispls[p], rdispls[p]);
  //   // }
  // }

   free(sendbuf_i);
   free(sendbuf_j);
   free(sendbuf_v);
  // free(recvbuf_j);
  // free(recvbuf_i);
  // free(recvbuf_v);
}

void ConstructVectors(DMEM_AllData *dmem_all_data,
                      hypre_ParCSRMatrix *A,
                      DMEM_VectorData *vector,
                      int fine_flag)
{
   void *object;
   HYPRE_Real *values;
   HYPRE_IJVector ij_u = NULL;
   HYPRE_IJVector ij_f = NULL;
   HYPRE_IJVector ij_x = NULL;
   HYPRE_IJVector ij_y = NULL;
   HYPRE_IJVector ij_z = NULL;
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

   values = hypre_CTAlloc(HYPRE_Real, local_num_cols, dmem_all_data->input.hypre_memory);
   srand(0);
   if (dmem_all_data->input.init_guess_type == INITGUESS_RAND){
      for (int i = 0; i < local_num_cols; i++) values[i] = RandDouble(-1.0, 1.0);
   } 
   else if (dmem_all_data->input.init_guess_type == INITGUESS_ONES){
      for (int i = 0; i < local_num_cols; i++) values[i] = 1.0;
   }     
   else {
      for (int i = 0; i < local_num_cols; i++) values[i] = 0.0;
   }

   /* initialize fine grid approximation to the solution */
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
   HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_x);
   HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
   HYPRE_IJVectorGetObject(ij_x, &object);
   vector->x = (HYPRE_ParVector)object;

   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_u);
   HYPRE_IJVectorSetObjectType(ij_u, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_u);
   HYPRE_IJVectorSetValues(ij_u, local_num_cols, NULL, values);
   HYPRE_IJVectorGetObject(ij_u, &object);
   vector->u = (HYPRE_ParVector)object;

   /* initialize correction vector */
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_e);
   for (HYPRE_Int i = 0; i < local_num_cols; i++) values[i] = 0.0;
   HYPRE_IJVectorSetObjectType(ij_e, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_e);
   HYPRE_IJVectorSetValues(ij_e, local_num_cols, NULL, values);
   HYPRE_IJVectorGetObject(ij_e, &object);
   vector->e = (HYPRE_ParVector)object;

  // HYPRE_IJVectorSetObjectType(ij_y, HYPRE_PARCSR);
  // HYPRE_IJVectorInitialize(ij_y);
  // HYPRE_IJVectorSetValues(ij_y, local_num_cols, NULL, values);
  // HYPRE_IJVectorGetObject(ij_y, &object);
  // vector->y = (HYPRE_ParVector)object;

  // HYPRE_IJVectorSetObjectType(ij_z, HYPRE_PARCSR);
  // HYPRE_IJVectorInitialize(ij_z);
  // HYPRE_IJVectorSetValues(ij_z, local_num_cols, NULL, values);
  // HYPRE_IJVectorGetObject(ij_z, &object);
  // vector->z = (HYPRE_ParVector)object;

   hypre_TFree(values, dmem_all_data->input.hypre_memory);

   values = hypre_CTAlloc(HYPRE_Real, local_num_rows, dmem_all_data->input.hypre_memory);

   if (dmem_all_data->input.rhs_type == RHS_ZEROS){
      for (int i = 0; i < local_num_rows; i++) values[i] = 0.0;
   }
   else if (dmem_all_data->input.rhs_type == RHS_ONES){
      for (int i = 0; i < local_num_rows; i++) values[i] = 1.0;
   }
   else {
      for (int i = 0; i < local_num_rows; i++) values[i] = RandDouble(-1.0, 1.0);
   }

   /* initialize fine grid right-hand side */
   if (dmem_all_data->input.rhs_type != RHS_FROM_PROBLEM || fine_flag == 0){
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b); 
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      HYPRE_IJVectorGetObject(ij_b, &object);
      vector->b = (HYPRE_ParVector)object;
   }
   else {
   }

   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_f);
   HYPRE_IJVectorSetObjectType(ij_f, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_f);
   HYPRE_IJVectorSetValues(ij_f, local_num_rows, NULL, values);
   HYPRE_IJVectorGetObject(ij_f, &object);
   vector->f = (HYPRE_ParVector)object;

   /* initialize fine grid residual */
   HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_r);
   HYPRE_IJVectorSetObjectType(ij_r, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_r);
   HYPRE_IJVectorSetValues(ij_r, local_num_rows, NULL, values);
   HYPRE_IJVectorGetObject(ij_r, &object);
   vector->r = (HYPRE_ParVector)object;

   hypre_TFree(values, dmem_all_data->input.hypre_memory);
}

void ResetVector(DMEM_AllData *dmem_all_data,
                 DMEM_VectorData *vector)
{
   hypre_ParVector *u = vector->u;
   hypre_ParVector *f = vector->f;
   hypre_ParVector *x = vector->x;
   hypre_ParVector *b = vector->b;
   hypre_ParVector *r = vector->r;
   hypre_ParVector *e = vector->e;
   hypre_Vector *u_local  = hypre_ParVectorLocalVector(u);
   hypre_Vector *f_local  = hypre_ParVectorLocalVector(f);
   hypre_Vector *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector *b_local  = hypre_ParVectorLocalVector(b);
   hypre_Vector *r_local  = hypre_ParVectorLocalVector(r);
   hypre_Vector *e_local  = hypre_ParVectorLocalVector(e);
   HYPRE_Real *u_local_data = hypre_VectorData(u_local);
   HYPRE_Real *f_local_data = hypre_VectorData(f_local);
   HYPRE_Real *x_local_data = hypre_VectorData(x_local);
   HYPRE_Real *b_local_data = hypre_VectorData(b_local);
   HYPRE_Real *r_local_data = hypre_VectorData(r_local);
   HYPRE_Real *e_local_data = hypre_VectorData(e_local);

   int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(dmem_all_data->matrix.A_fine));
   srand(0);
   if (dmem_all_data->input.init_guess_type == INITGUESS_RAND){
      for (int i = 0; i < num_rows; i++){
         x_local_data[i] = RandDouble(-1.0, 1.0);
      }
   }
   else if (dmem_all_data->input.init_guess_type == INITGUESS_ONES){
      DMEM_HypreParVector_Set(x, 1.0, num_rows);
   }
   else {
      DMEM_HypreParVector_Set(x, 0.0, num_rows);
   }

  // if (dmem_all_data->input.rhs_type == RHS_ZEROS){
  //    DMEM_HypreParVector_Set(b, 0.0, num_rows);
  //    DMEM_HypreParVector_Set(f, 0.0, num_rows);
  // }
  // else if (dmem_all_data->input.rhs_type == RHS_ONES){
  //    DMEM_HypreParVector_Set(b, 1.0, num_rows);
  //    DMEM_HypreParVector_Set(f, 1.0, num_rows);
  // }
  // else {
  //   // b_local_data[i] = f_local_data[i] = RandDouble(-1.0, 1.0);
  // }
   DMEM_HypreParVector_Set(e, 0.0, num_rows);

  // e_local_data[i] = 0.0;
  // for (int i = 0; i < num_rows; i++){
  //    if (dmem_all_data->input.init_guess_type == INITGUESS_RAND){
  //       x_local_data[i] = u_local_data[i] = RandDouble(-1.0, 1.0);
  //    }
  //    else if (dmem_all_data->input.init_guess_type == INITGUESS_ONES){
  //       x_local_data[i] = u_local_data[i] = 1.0;
  //    }
  //    else {
  //       x_local_data[i] = u_local_data[i] = 0.0;
  //    }

  //    if (dmem_all_data->input.rhs_type == RHS_ZEROS){
  //       b_local_data[i] = f_local_data[i] = 0.0;
  //    }
  //    else if (dmem_all_data->input.rhs_type == RHS_ONES){
  //       b_local_data[i] = f_local_data[i] = 1.0;
  //    }
  //    else {
  //       b_local_data[i] = f_local_data[i] = RandDouble(-1.0, 1.0);
  //    }

  //    e_local_data[i] = 0.0;
  // }
}

void ResetNorms(DMEM_AllData *dmem_all_data,
                     DMEM_VectorData *vector)
{
   hypre_ParVector *u = vector->u;
   hypre_ParVector *f = vector->f;
   hypre_ParVector *x = vector->x;
   hypre_ParVector *b = vector->b;
   hypre_ParVector *r = vector->r;
   hypre_ParVector *e = vector->e;

   hypre_ParAMGData *amg_data_fine = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParVector *Vtemp = hypre_ParAMGDataVtemp(amg_data_fine);

   hypre_ParCSRMatrixMatvecOutOfPlace(-1.0,
                                      dmem_all_data->matrix.A_fine,
                                      x,
                                      1.0,
                                      b,
                                      r);
   dmem_all_data->output.r0_norm2 = sqrt(hypre_ParVectorInnerProd(r, r));

   hypre_ParCSRMatrixMatvecOutOfPlace(1.0,
                                      dmem_all_data->matrix.A_fine,
                                      x,
                                      0.0,
                                      Vtemp,
                                      e);

   hypre_ParVector *Ax = e;
   dmem_all_data->output.e0_Anorm = sqrt(hypre_ParVectorInnerProd(Ax, x));

   
}

void ResetOutputData(DMEM_AllData *dmem_all_data)
{
   dmem_all_data->output.level_wtime.resize(dmem_all_data->grid.num_levels, 0.0);
   dmem_all_data->output.solve_wtime = 0.0;
   dmem_all_data->output.residual_wtime = 0.0;
   dmem_all_data->output.residual_norm_wtime = 0.0;
   dmem_all_data->output.restrict_wtime = 0.0;
   dmem_all_data->output.prolong_wtime = 0.0;
   dmem_all_data->output.smooth_wtime = 0.0;
   dmem_all_data->output.correct_wtime = 0.0;
   dmem_all_data->output.comm_wtime = 0.0;
   dmem_all_data->output.comp_wtime = 0.0;
   dmem_all_data->output.coarsest_solve_wtime = 0.0;
   dmem_all_data->output.start_wtime = 0.0;
   dmem_all_data->output.end_wtime = 0.0;
   dmem_all_data->output.inner_solve_wtime = 0.0;
   dmem_all_data->output.mpiisend_wtime = 0.0;
   dmem_all_data->output.mpiirecv_wtime = 0.0;
   dmem_all_data->output.mpitest_wtime = 0.0;
   dmem_all_data->output.mpiwait_wtime = 0.0;
}

void ResetCommData(DMEM_CommData *comm_data)
{
   if (comm_data->type == GRIDK_OUTSIDE_SEND || comm_data->type == FINE_INTRA_OUTSIDE_SEND){
      for (int i = 0; i < comm_data->procs.size(); i++){
         comm_data->num_inflight[i] = 0;
         comm_data->next_inflight[i] = 0;
         comm_data->message_count[i] = 0;
         comm_data->done_flags[i] = 0;
         for (int j = 0; j < comm_data->max_inflight[i]; j++){
            comm_data->requests_inflight[i][j] = MPI_REQUEST_NULL;
            comm_data->inflight_flags[i][j] = 0;
            for (int k = 0; k < comm_data->len[i]+1; k++){
               comm_data->data_inflight[i][j][k] = 0.0;
            }
         }
      }
   }
  // else {
      for (int i = 0; i < comm_data->procs.size(); i++){
         comm_data->requests[i] = MPI_REQUEST_NULL;
         comm_data->message_count[i] = 0;
         comm_data->done_flags[i] = 0;
         for (int j = 0; j < comm_data->len[i]+1; j++){
            comm_data->data[i][j] = 0;
         }
      }
  // }
}

void DMEM_ResetAllCommData(DMEM_AllData *dmem_all_data)
{
   if (dmem_all_data->input.solver == MULTADD ||
       dmem_all_data->input.solver == BPX ||
       dmem_all_data->input.solver == MULT_MULTADD ||
       dmem_all_data->input.solver == AFACX){
      ResetCommData(&(dmem_all_data->comm.finestToGridk_Residual_insideSend));
      ResetCommData(&(dmem_all_data->comm.finestToGridk_Residual_insideRecv));
      ResetCommData(&(dmem_all_data->comm.finestToGridk_Correct_insideSend));
      ResetCommData(&(dmem_all_data->comm.finestToGridk_Correct_insideRecv));

      if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
         ResetCommData(&(dmem_all_data->comm.finestToGridk_Residual_outsideSend));
         ResetCommData(&(dmem_all_data->comm.finestToGridk_Residual_outsideRecv));
         ResetCommData(&(dmem_all_data->comm.finestToGridk_Correct_outsideSend));
         ResetCommData(&(dmem_all_data->comm.finestToGridk_Correct_outsideRecv));

         ResetCommData(&(dmem_all_data->comm.finestIntra_insideSend));
         ResetCommData(&(dmem_all_data->comm.finestIntra_insideRecv));
         ResetCommData(&(dmem_all_data->comm.finestIntra_outsideSend));
         ResetCommData(&(dmem_all_data->comm.finestIntra_outsideRecv));


        // hypre_ParCSRMatrix *A = dmem_all_data->matrix.A_fine;
        // hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
        // hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);

        // HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
        // for (int i = 0; i < hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends)+1; i++){
        //    dmem_all_data->comm.fine_send_data[i] = 0;
        // }
        // HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
        // HYPRE_Real *x_ghost_data = hypre_VectorData(dmem_all_data->vector_fine.x_ghost);
        // for (int i = 0; i < num_cols_offd+1; i++){
        //    dmem_all_data->comm.fine_recv_data[i] = 0;
        //    x_ghost_data[i] = 0;
        // }
      }
      else {
         ResetCommData(&(dmem_all_data->comm.gridjToGridk_Correct_insideSend));
         ResetCommData(&(dmem_all_data->comm.gridjToGridk_Correct_outsideSend));
         ResetCommData(&(dmem_all_data->comm.gridjToGridk_Correct_insideRecv));
         ResetCommData(&(dmem_all_data->comm.gridjToGridk_Correct_outsideRecv));

         ResetCommData(&(dmem_all_data->comm.finestIntra_outsideSend));
         ResetCommData(&(dmem_all_data->comm.finestIntra_outsideRecv));
      }
   }
}

void DMEM_ResetData(DMEM_AllData *dmem_all_data)
{
   int my_grid = dmem_all_data->grid.my_grid;

   DMEM_ResetAllCommData(dmem_all_data);
   ResetOutputData(dmem_all_data);
   ResetVector(dmem_all_data, &(dmem_all_data->vector_fine));
   ResetNorms(dmem_all_data, &(dmem_all_data->vector_fine));

   dmem_all_data->comm.is_async_smoothing_flag = 0;

   if (dmem_all_data->input.async_smoother_flag == 1 &&
       (dmem_all_data->input.solver == MULTADD ||
        dmem_all_data->input.solver == MULT_MULTADD)){
      if (my_grid == 0){
         if (dmem_all_data->input.async_flag == 1){
            dmem_all_data->comm.finestIntra_outsideSend.type = FINE_INTRA_INSIDE_SEND;
            dmem_all_data->comm.finestIntra_outsideRecv.type = FINE_INTRA_INSIDE_RECV;
         }

         HYPRE_Real *x_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.x));
         HYPRE_Real *x_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.x_ghost);
         SendRecv(dmem_all_data,
                  &(dmem_all_data->comm.finestIntra_outsideSend),
                  x_local_data,
                  WRITE);
         SendRecv(dmem_all_data,
                  &(dmem_all_data->comm.finestIntra_outsideRecv),
                  x_ghost_data,
                  READ);
         CompleteRecv(dmem_all_data,
                      &(dmem_all_data->comm.finestIntra_outsideRecv),
                      x_ghost_data,
                      READ);
         hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
                           dmem_all_data->comm.finestIntra_outsideSend.requests,
                           MPI_STATUSES_IGNORE);

        // HYPRE_Real *b_local_data = hypre_VectorData(hypre_ParVectorLocalVector(dmem_all_data->vector_gridk.b));
        // HYPRE_Real *b_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.b_ghost);
        // SendRecv(dmem_all_data,
        //          &(dmem_all_data->comm.finestIntra_outsideSend),
        //          b_local_data,
        //          WRITE);
        // SendRecv(dmem_all_data,
        //          &(dmem_all_data->comm.finestIntra_outsideRecv),
        //          b_ghost_data,
        //          READ);
        // CompleteRecv(dmem_all_data,
        //              &(dmem_all_data->comm.finestIntra_outsideRecv),
        //              b_ghost_data,
        //              READ);
        // hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
        //                   dmem_all_data->comm.finestIntra_outsideSend.requests,
        //                   MPI_STATUSES_IGNORE);

        // HYPRE_Real *a_diag_data = hypre_VectorData(dmem_all_data->vector_gridk.a_diag);
        // HYPRE_Real *a_diag_ghost_data = hypre_VectorData(dmem_all_data->vector_gridk.a_diag_ghost);
        // SendRecv(dmem_all_data,
        //          &(dmem_all_data->comm.finestIntra_outsideSend),
        //          a_diag_data,
        //          WRITE);
        // SendRecv(dmem_all_data,
        //          &(dmem_all_data->comm.finestIntra_outsideRecv),
        //          a_diag_ghost_data,
        //          READ);
        // CompleteRecv(dmem_all_data,
        //              &(dmem_all_data->comm.finestIntra_outsideRecv),
        //              a_diag_ghost_data,
        //              READ);
        // hypre_MPI_Waitall(dmem_all_data->comm.finestIntra_outsideSend.procs.size(),
        //                   dmem_all_data->comm.finestIntra_outsideSend.requests,
        //                   MPI_STATUSES_IGNORE);

         if (dmem_all_data->input.async_flag == 1){
            dmem_all_data->comm.finestIntra_outsideSend.type = FINE_INTRA_OUTSIDE_SEND;
            dmem_all_data->comm.finestIntra_outsideRecv.type = FINE_INTRA_OUTSIDE_RECV;
         }
      }
   }   
}

void AssignProcs(DMEM_AllData *dmem_all_data)
{
   int num_levels = dmem_all_data->grid.num_levels;
   int num_procs, my_id;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   dmem_all_data->grid.procs = (int **)calloc(num_levels, sizeof(int *));
   dmem_all_data->grid.num_procs_level = (int *)calloc(num_levels, sizeof(int));

  // dmem_all_data->grid.my_grid = -1;
  // int procs_per_level = num_procs/num_levels;
  // int extra = num_procs - procs_per_level*(num_levels - 1);
  // for (int level = 0; level < num_levels-1; level++){
  //    dmem_all_data->grid.num_procs_level[level] = procs_per_level;
  // }
  // dmem_all_data->grid.num_procs_level[num_levels-1] = extra;
  // int count_id = 0;
  // for (int level = 0; level < num_levels; level++){
  //    dmem_all_data->grid.procs[level] = 
  //       (int *)calloc(dmem_all_data->grid.num_procs_level[level], sizeof(int));
  //    for (int i = 0; i < dmem_all_data->grid.num_procs_level[level]; i++){
  //       dmem_all_data->grid.procs[level][i] = count_id;
  //       if (my_id == count_id){
  //          dmem_all_data->grid.my_grid = level;
  //       }
  //       count_id++;
  //    }
  // }


   int proc_id = 0;
   int finest_level;
   int count_num_procs = num_procs;
   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      finest_level = dmem_all_data->input.coarsest_mult_level + 1;
   }
   else {
      finest_level = dmem_all_data->input.coarsest_mult_level;
   }
   for (int level = finest_level; level < num_levels; level++){
      int current_candidate_procs;
      if (level == num_levels-1 || count_num_procs == 1){
         current_candidate_procs = count_num_procs;
      }
      else {
         if (dmem_all_data->input.assign_procs_type == ASSIGN_PROCS_SCALAR){
            current_candidate_procs = max((int)floor(((double)current_candidate_procs)*dmem_all_data->input.assign_procs_scalar), 1);
         }
         else if (count_num_procs == num_levels - level){
            current_candidate_procs = 1;
         }
         else {
            current_candidate_procs = max((int)ceil(dmem_all_data->grid.frac_level_work[level] * (double)num_procs), 1);
           // if (level == 0){
           //    current_candidate_procs += 10;
           // }
           // else {
               while (1){
                  int next_candidate_procs = current_candidate_procs-1;
                  double next_candidate_procs_frac = (double)next_candidate_procs/(double)num_procs;
                  double diff_current = fabs(dmem_all_data->grid.frac_level_work[level] - (double)current_candidate_procs/(double)num_procs);
                  double diff_next_candidate_procs = fabs(dmem_all_data->grid.frac_level_work[level] - next_candidate_procs_frac);
                 // if (count_num_procs - current_candidate_procs <= num_levels - level){
                 //    current_candidate_procs = count_num_procs  - (num_levels - level);
                 //    break;
                 // }
                  if ((diff_current <= diff_next_candidate_procs) || current_candidate_procs == 1){
                     break;
                  }
                  current_candidate_procs--;
               }
           // }
   
           // current_candidate_procs = max((int)floor(dmem_all_data->grid.frac_level_work[level] * (double)num_procs), 1);
           // while (1){
           //    int next_candidate_procs = current_candidate_procs+1;
           //    double next_candidate_procs_frac = (double)next_candidate_procs/(double)num_procs;
           //    double diff_current = fabs(dmem_all_data->grid.frac_level_work[level] - (double)current_candidate_procs/(double)num_procs);
           //    double diff_next_candidate_procs = fabs(dmem_all_data->grid.frac_level_work[level] - next_candidate_procs_frac);
           //    if ((diff_current <= diff_next_candidate_procs) || (next_candidate_procs_frac > dmem_all_data->grid.frac_level_work[level])){
           //       break;
           //    }
           //    current_candidate_procs++;
           // }
            }
      }

      dmem_all_data->grid.num_procs_level[level] = current_candidate_procs;
      dmem_all_data->grid.procs[level] = (int *)calloc(dmem_all_data->grid.num_procs_level[level], sizeof(int));
      for (int p = proc_id, i = 0; p < proc_id + current_candidate_procs; p++, i++){
         if (my_id == p){
            dmem_all_data->grid.my_grid = level;
         }
         dmem_all_data->grid.procs[level][i] = p;
      }
      count_num_procs -= current_candidate_procs;
      proc_id += current_candidate_procs;
   }

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //       printf("id %d, grid %d\n", my_id, dmem_all_data->grid.my_grid);
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }
  // if (my_id == 0){
  //    for (int level = 0; level < num_levels; level++){
  //       printf("level %d, num %d, work %f, %f\n", level, dmem_all_data->grid.num_procs_level[level], dmem_all_data->grid.frac_level_work[level], (double)dmem_all_data->grid.num_procs_level[level]/(double)num_procs);
  //       for (int i = 0; i < dmem_all_data->grid.num_procs_level[level]; i++){
  //          printf("\t%d\n", dmem_all_data->grid.procs[level][i]);
  //       }
  //    }
  // }
  // MPI_Barrier(MPI_COMM_WORLD);
   MPI_Comm_split(MPI_COMM_WORLD,
                  dmem_all_data->grid.my_grid,
                  my_id,
                  &(dmem_all_data->grid.my_comm));
  // printf("%d, %d\n", my_id, dmem_all_data->grid.my_grid);
}

//TODO: revisit how work is approximated
void ComputeWork(DMEM_AllData *dmem_all_data)
{
   int coarsest_level;
   int fine_grid, coarse_grid;
   int num_procs, my_id;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   hypre_ParAMGData *amg_data = (hypre_ParAMGData *)dmem_all_data->hypre.solver;
   hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **P_array = hypre_ParAMGDataAArray(amg_data);
   hypre_ParCSRMatrix **R_array = hypre_ParAMGDataAArray(amg_data);

   dmem_all_data->grid.level_work = (double *)calloc(dmem_all_data->grid.num_levels, sizeof(double));

   int finest_level;
   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      finest_level = dmem_all_data->input.coarsest_mult_level+1;
   }
   else {
      finest_level = dmem_all_data->input.coarsest_mult_level;
   }

   for (int level = finest_level; level < dmem_all_data->grid.num_levels; level++){
      if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
         dmem_all_data->grid.level_work[level] = (double)(hypre_ParCSRMatrixNumNonzeros(A_array[0]) + hypre_ParCSRMatrixGlobalNumRows(A_array[0])) / (double)dmem_all_data->grid.num_levels + (double)hypre_ParCSRMatrixGlobalNumRows(A_array[0]);
      }
      else {
         dmem_all_data->grid.level_work[level] = (double)(hypre_ParCSRMatrixNumNonzeros(A_array[0]) + (double)((dmem_all_data->grid.num_levels + 2)*hypre_ParCSRMatrixGlobalNumRows(A_array[0])));
      }

      if (dmem_all_data->input.solver == MULTADD ||
          dmem_all_data->input.solver == BPX ||
          dmem_all_data->input.solver == MULT_MULTADD){
         coarsest_level = level;
      }
      else if (dmem_all_data->input.solver == AFACX){
         coarsest_level = level+1;
      }

      if (dmem_all_data->input.num_interpolants == ONE_INTERPOLANT && level > 0){
         dmem_all_data->grid.level_work[level] += (double)hypre_ParCSRMatrixNumNonzeros(dmem_all_data->matrix.P_fine[level-1]);
      }
      else {
         for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
            fine_grid = inner_level;
            coarse_grid = inner_level + 1;
            dmem_all_data->grid.level_work[level] += (double)hypre_ParCSRMatrixNumNonzeros(R_array[fine_grid]);
         }
      }

      fine_grid = level;
      coarse_grid = level + 1;
      if (level == dmem_all_data->grid.num_levels-1){
         dmem_all_data->grid.level_work[level] += pow((double)(hypre_ParCSRMatrixGlobalNumRows(A_array[fine_grid])), 2.0);
      }
      else {
         if (level == 0 && dmem_all_data->input.solver == MULTADD && 
             (dmem_all_data->input.smoother == ASYNC_JACOBI ||
              dmem_all_data->input.smoother == ASYNC_HYBRID_JACOBI_GAUSS_SEIDEL ||
              dmem_all_data->input.smoother == ASYNC_STOCHASTIC_PARALLEL_SOUTHWELL)){
           // dmem_all_data->grid.level_work[level] += (double)hypre_ParCSRMatrixGlobalNumRows(A_array[fine_grid]);
            dmem_all_data->grid.level_work[level] += (double)(hypre_ParCSRMatrixNumNonzeros(A_array[fine_grid]) + 2.0*(double)hypre_ParCSRMatrixGlobalNumRows(A_array[fine_grid]));
         }
         else {
            if (dmem_all_data->input.solver == MULTADD ||
                dmem_all_data->input.solver == BPX ||
                dmem_all_data->input.solver == MULT_MULTADD){
              // dmem_all_data->grid.level_work[level] += (double)hypre_ParCSRMatrixGlobalNumRows(A_array[fine_grid]);
               dmem_all_data->grid.level_work[level] += (double)(hypre_ParCSRMatrixNumNonzeros(A_array[fine_grid]) + 2.0*(double)hypre_ParCSRMatrixGlobalNumRows(A_array[fine_grid]));
            }
            else if (dmem_all_data->input.solver == AFACX){
               dmem_all_data->grid.level_work[level] +=
                  (double)((dmem_all_data->input.num_coarse_smooth_sweeps-1) * hypre_ParCSRMatrixNumNonzeros(A_array[coarse_grid]) +
                           dmem_all_data->input.num_coarse_smooth_sweeps * hypre_ParCSRMatrixGlobalNumRows(A_array[coarse_grid]) +
                           hypre_ParCSRMatrixNumNonzeros(P_array[fine_grid]) +
                           hypre_ParCSRMatrixNumNonzeros(A_array[fine_grid]) +
                           (dmem_all_data->input.num_fine_smooth_sweeps-1) * hypre_ParCSRMatrixNumNonzeros(A_array[fine_grid]) +
                           dmem_all_data->input.num_fine_smooth_sweeps * hypre_ParCSRMatrixGlobalNumRows(A_array[fine_grid]));
            }
         }
      }

      if (dmem_all_data->input.num_interpolants == ONE_INTERPOLANT && level > 0){
         dmem_all_data->grid.level_work[level] += (double)hypre_ParCSRMatrixNumNonzeros(dmem_all_data->matrix.P_fine[level-1]);
      }
      else {
         coarsest_level = level;
         for (int inner_level = 0; inner_level < coarsest_level; inner_level++){
            fine_grid = inner_level;
            coarse_grid = inner_level + 1;
            dmem_all_data->grid.level_work[level] += (double)hypre_ParCSRMatrixNumNonzeros(P_array[fine_grid]);
         }
      }
   }
  // dmem_all_data->grid.level_work[0] = 1.0;
   dmem_all_data->grid.tot_work = 0.0;
   dmem_all_data->grid.frac_level_work = (double *)calloc(dmem_all_data->grid.num_levels, sizeof(double));
   for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
      dmem_all_data->grid.tot_work += dmem_all_data->grid.level_work[level];
   }
   for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
      dmem_all_data->grid.frac_level_work[level] = dmem_all_data->grid.level_work[level] / dmem_all_data->grid.tot_work;
   }

}

void BuildMatrix(DMEM_AllData *dmem_all_data, HYPRE_ParCSRMatrix *A, HYPRE_ParVector *rhs, MPI_Comm comm)
{
   if (dmem_all_data->input.test_problem == MFEM_ELAST ||
       dmem_all_data->input.test_problem == MFEM_ELAST_AMR){
      DMEM_BuildMfemMatrix(dmem_all_data,
                           A,
                           rhs,
                           comm);
   }
   else {
      DMEM_BuildHypreMatrix(dmem_all_data,
                            A,
                            rhs,
                            comm,
                            dmem_all_data->matrix.nx, dmem_all_data->matrix.ny, dmem_all_data->matrix.nz,
                            dmem_all_data->matrix.difconv_cx, dmem_all_data->matrix.difconv_cy, dmem_all_data->matrix.difconv_cz,
                            dmem_all_data->matrix.difconv_ax, dmem_all_data->matrix.difconv_ay, dmem_all_data->matrix.difconv_az,
                            dmem_all_data->matrix.vardifconv_eps,
                            dmem_all_data->matrix.difconv_atype);
   }
  // DMEM_ParMfem(dmem_all_data, A, comm);
}


void SetVectorComms(DMEM_AllData *dmem_all_data,
                    DMEM_VectorData *vector,
                    MPI_Comm comm)
{
  // hypre_ParVector *x = vector->x;
  // hypre_ParVector *b = vector->b;
  // hypre_ParVector *r = vector->r;
  // hypre_ParVector *e = vector->e;

  // hypre_ParVectorComm(x) = comm;
  // hypre_ParVectorComm(b) = comm;
  // hypre_ParVectorComm(r) = comm;
  // hypre_ParVectorComm(e) = comm;
}
