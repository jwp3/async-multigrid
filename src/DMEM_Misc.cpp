#include "DMEM_Main.hpp"
#include "DMEM_Misc.hpp"
#include "_hypre_utilities.h"

extern HYPRE_Int hypre_memory;

void DMEM_PrintOutput(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
 
   int sum_cycles, min_cycles, max_cycles;
   int sum_relax, min_relax, max_relax;
   double mean_cycles;
   double mean_level_wtime;
   double min_solve_wtime, max_solve_wtime, mean_solve_wtime;
   double min_residual_wtime, max_residual_wtime, mean_residual_wtime;
   double min_residual_norm_wtime, max_residual_norm_wtime, mean_residual_norm_wtime;
   double min_prolong_wtime, max_prolong_wtime, mean_prolong_wtime;
   double min_restrict_wtime, max_restrict_wtime, mean_restrict_wtime;
   double min_smooth_wtime, max_smooth_wtime, mean_smooth_wtime;
   double min_matvec_wtime, max_matvec_wtime, mean_matvec_wtime;
   double min_vecop_wtime, max_vecop_wtime, mean_vecop_wtime;
   double min_coarsest_solve_wtime, max_coarsest_solve_wtime, mean_coarsest_solve_wtime;
   double min_comm_wtime, max_comm_wtime, mean_comm_wtime;
   double min_start_wtime, max_start_wtime, mean_start_wtime;
   double min_end_wtime, max_end_wtime, mean_end_wtime;
   double min_mpiisend_wtime, max_mpiisend_wtime, mean_mpiisend_wtime;
   double min_mpiirecv_wtime, max_mpiirecv_wtime, mean_mpiirecv_wtime;
   double min_mpiwait_wtime, max_mpiwait_wtime, mean_mpiwait_wtime;
   double min_mpitest_wtime, max_mpitest_wtime, mean_mpitest_wtime;
   
   int sum_num_messages, min_num_messages, max_num_messages;

   double solve_wtime, residual_wtime, residual_norm_wtime, prolong_wtime, restrict_wtime, vecop_wtime, matvec_wtime, smooth_wtime, coarsest_solve_wtime, comm_wtime, start_wtime, end_wtime;

  // if (dmem_all_data->input.solver == MULTADD){
  //    int my_id_local, num_procs_local;
  //    MPI_Comm_rank(dmem_all_data->grid.my_comm, &my_id_local);
  //    MPI_Comm_size(dmem_all_data->grid.my_comm, &num_procs_local);
  //    for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
  //       if (level == dmem_all_data->grid.my_grid){
  //          if (my_id_local == 0){
  //         // if (level == 0){
  //             printf("level %d num cycles %d smooth %e work %e\n", level, dmem_all_data->iter.cycle, dmem_all_data->output.smooth_wtime, dmem_all_data->grid.level_work[level]);
  //          }
  //       }
  //       MPI_Barrier(MPI_COMM_WORLD);
  //    }
  //    MPI_Barrier(MPI_COMM_WORLD);
  // }

   hypre_ParVector *r = dmem_all_data->vector_fine.r;
   dmem_all_data->output.r_norm2 = sqrt(hypre_ParVectorInnerProd(r, r));

   hypre_ParVector *Ax = dmem_all_data->vector_fine.e;
   hypre_ParVector *x = dmem_all_data->vector_fine.x;
   dmem_all_data->output.e_Anorm = sqrt(hypre_ParVectorInnerProd(Ax, x));

   MPI_Reduce(&(dmem_all_data->output.solve_wtime),          &mean_solve_wtime,          1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.residual_wtime),       &mean_residual_wtime,       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.residual_norm_wtime),  &mean_residual_norm_wtime,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.restrict_wtime),       &mean_restrict_wtime,       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.prolong_wtime),        &mean_prolong_wtime,        1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.matvec_wtime),         &mean_matvec_wtime,         1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.vecop_wtime),          &mean_vecop_wtime,          1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.smooth_wtime),         &mean_smooth_wtime,         1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  // MPI_Reduce(&(dmem_all_data->output.coarsest_solve_wtime), &mean_coarsest_solve_wtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.comm_wtime),           &mean_comm_wtime,           1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.start_wtime),          &mean_start_wtime,          1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.end_wtime),            &mean_end_wtime,            1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiisend_wtime),       &mean_mpiisend_wtime,       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiirecv_wtime),       &mean_mpiirecv_wtime,       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpitest_wtime),        &mean_mpitest_wtime,        1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiwait_wtime),        &mean_mpiwait_wtime,        1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->iter.cycle),                  &sum_cycles,                1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->iter.relax),                  &sum_relax,                 1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.num_messages),         &sum_num_messages,          1, MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);
   

   mean_solve_wtime /= (double)num_procs;
   mean_residual_wtime /= (double)num_procs;
   mean_residual_norm_wtime /= (double)num_procs;
   if (dmem_all_data->input.solver == MULTADD ||
       dmem_all_data->input.solver == BPX ||
       dmem_all_data->input.solver == AFACX){
      mean_restrict_wtime /= (double)(num_procs - dmem_all_data->grid.num_procs_level[0]);
      mean_prolong_wtime /= (double)(num_procs - dmem_all_data->grid.num_procs_level[0]);
   }
   else {
      mean_restrict_wtime /= (double)(num_procs);
      mean_prolong_wtime /= (double)(num_procs);
   }
   mean_smooth_wtime /= (double)num_procs;
   mean_matvec_wtime /= (double)num_procs;
   mean_vecop_wtime /= (double)num_procs;
   mean_comm_wtime /= (double)num_procs;
   mean_start_wtime /= (double)num_procs;
   mean_end_wtime /= (double)num_procs;
   mean_mpiisend_wtime /= (double)num_procs;
   mean_mpiirecv_wtime /= (double)num_procs;
   mean_mpitest_wtime /= (double)num_procs;
   mean_mpiwait_wtime /= (double)num_procs;
   mean_cycles = (double)sum_cycles/(double)num_procs;

   if ((dmem_all_data->input.solver == MULTADD ||
       dmem_all_data->input.solver == BPX) &&
       dmem_all_data->grid.my_grid == 0){
      dmem_all_data->output.restrict_wtime = DBL_MAX;
      dmem_all_data->output.prolong_wtime = DBL_MAX;
   }

   MPI_Reduce(&(dmem_all_data->output.solve_wtime),          &min_solve_wtime,           1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.residual_wtime),       &min_residual_wtime,        1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.residual_norm_wtime),  &min_residual_norm_wtime,   1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.restrict_wtime),       &min_restrict_wtime,        1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.prolong_wtime),        &min_prolong_wtime,         1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.matvec_wtime),         &min_matvec_wtime,          1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.vecop_wtime),          &min_vecop_wtime,           1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.smooth_wtime),         &min_smooth_wtime,          1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  // MPI_Reduce(&(dmem_all_data->output.coarsest_solve_wtime), &min_coarsest_solve_wtime,  1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.comm_wtime),           &min_comm_wtime,            1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.start_wtime),          &min_start_wtime,           1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.end_wtime),            &min_end_wtime,             1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiisend_wtime),       &min_mpiisend_wtime,        1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiirecv_wtime),       &min_mpiirecv_wtime,        1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpitest_wtime),        &min_mpitest_wtime,         1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiwait_wtime),        &min_mpiwait_wtime,         1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->iter.cycle),                  &min_cycles,                1, MPI_INT,    MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.num_messages),         &min_num_messages,          1, MPI_INT,    MPI_MIN, 0, MPI_COMM_WORLD);

   if ((dmem_all_data->input.solver == MULTADD ||
        dmem_all_data->input.solver == BPX) && 
        dmem_all_data->grid.my_grid == 0){
      dmem_all_data->output.restrict_wtime = 0.0;
      dmem_all_data->output.prolong_wtime = 0.0;
   }
 
   MPI_Reduce(&(dmem_all_data->output.solve_wtime),          &max_solve_wtime,           1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.residual_wtime),       &max_residual_wtime,        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.residual_norm_wtime),  &max_residual_norm_wtime,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.restrict_wtime),       &max_restrict_wtime,        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.prolong_wtime),        &max_prolong_wtime,         1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.matvec_wtime),         &max_matvec_wtime,          1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.vecop_wtime),          &max_vecop_wtime,           1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.smooth_wtime),         &max_smooth_wtime,          1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.comm_wtime),           &max_comm_wtime,            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.start_wtime),          &max_start_wtime,           1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.end_wtime),            &max_end_wtime,             1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiisend_wtime),       &max_mpiisend_wtime,        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiirecv_wtime),       &max_mpiirecv_wtime,        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpitest_wtime),        &max_mpitest_wtime,         1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.mpiwait_wtime),        &max_mpiwait_wtime,         1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->iter.cycle),                  &max_cycles,                1, MPI_INT,    MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.num_messages),         &max_num_messages,          1, MPI_INT,    MPI_MAX, 0, MPI_COMM_WORLD);


  // if (dmem_all_data->input.solver == MULT){
  //    for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
  //       MPI_Reduce(&(dmem_all_data->output.level_wtime[level]), &mean_level_wtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  //       mean_level_wtime /= (double)num_procs;
  //       if (my_id == 0){
  //          printf("%d %e\n", level, mean_level_wtime);
  //       }
  //    }
  // }
  // else {
  //    int my_id_local, num_procs_local;
  //    MPI_Comm_rank(dmem_all_data->grid.my_comm, &my_id_local);
  //    MPI_Comm_size(dmem_all_data->grid.my_comm, &num_procs_local);
  //    if (dmem_all_data->grid.my_grid == dmem_all_data->grid.num_levels-1){
  //       for (int level = 0; level < dmem_all_data->grid.num_levels; level++){
  //          MPI_Reduce(&(dmem_all_data->output.level_wtime[level]), &mean_level_wtime, 1, MPI_DOUBLE, MPI_SUM, 0, dmem_all_data->grid.my_comm);
  //          mean_level_wtime /= (double)num_procs_local;
  //          if (my_id_local == 0){
  //             printf("%d %e\n", level, mean_level_wtime);
  //          }
  //       }
  //    }
  // }
  // MPI_Barrier(MPI_COMM_WORLD);

   if (my_id == 0){
      char print_str[1000];
      if (dmem_all_data->input.oneline_output_flag == 0){
         printf("Setup time = %e, build matrix time = %e\n\n", 
                dmem_all_data->output.setup_wtime, dmem_all_data->output.build_matrix_wtime);
         strcpy(print_str, "Relative Residual 2-norm = %e\n"
                           "Relative Error A-norm (only for rhs==0) = %e\n\n"
                           //"Setup stats\n\n"
                           "Solve stats          \t  mean  \t   max  \t   min  \n"
                           "---------------------\t--------\t--------\t--------\n"
                           "Cycles               \t%.2f\t\t%d\t\t%d\n"
                           "Solve time           \t%.2e\t%.2e\t%.2e\n"
                           "Residual time        \t%.2e\t%.2e\t%.2e\n"
                           "Residual norm time   \t%.2e\t%.2e\t%.2e\n"
                           "Prolong time         \t%.2e\t%.2e\t%.2e\n"
                           "Restrict time        \t%.2e\t%.2e\t%.2e\n"
                           "Matvec time          \t%.2e\t%.2e\t%.2e\n"
                           "Vecop time           \t%.2e\t%.2e\t%.2e\n"
                           "Smooth time          \t%.2e\t%.2e\t%.2e\n"
                           "Comm time            \t%.2e\t%.2e\t%.2e\n"
                           "Start time           \t%.2e\t%.2e\t%.2e\n"
                           "End time             \t%.2e\t%.2e\t%.2e\n"
                           "MPI_Isend time       \t%.2e\t%.2e\t%.2e\n"
                           "MPI_Irecv time       \t%.2e\t%.2e\t%.2e\n"
                           "MPI_Test time        \t%.2e\t%.2e\t%.2e\n"
                           "MPI_Wait time        \t%.2e\t%.2e\t%.2e\n"
                           "num messages         \t%.2d\t\t%.2d\t\t%.2d\n"
                           );
      }
      else {
         strcpy(print_str, "%e %e "
                           "%f %d %d "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%e %e %e "
                           "%d %d %d\n");
      }
      printf(print_str,
             dmem_all_data->output.r_norm2/dmem_all_data->output.r0_norm2,
             dmem_all_data->output.e_Anorm/dmem_all_data->output.e0_Anorm,
             mean_cycles, max_cycles, min_cycles,
             mean_solve_wtime, max_solve_wtime, min_solve_wtime,
             mean_residual_wtime, max_residual_wtime, min_residual_wtime,
             mean_residual_norm_wtime, max_residual_norm_wtime, min_residual_norm_wtime,
             mean_prolong_wtime, max_prolong_wtime, min_prolong_wtime,
             mean_restrict_wtime, max_restrict_wtime, min_restrict_wtime,
             mean_matvec_wtime, max_matvec_wtime, min_matvec_wtime,
             mean_vecop_wtime, max_vecop_wtime, min_vecop_wtime,
             mean_smooth_wtime, max_smooth_wtime, min_smooth_wtime,
             mean_comm_wtime, max_comm_wtime, min_comm_wtime,
             mean_start_wtime, max_start_wtime, min_start_wtime,
             mean_end_wtime, max_end_wtime, min_start_wtime,
             mean_mpiisend_wtime, max_mpiisend_wtime, min_mpiisend_wtime,
             mean_mpiirecv_wtime, max_mpiirecv_wtime, min_mpiirecv_wtime,
             mean_mpitest_wtime, max_mpitest_wtime, min_mpitest_wtime,
             mean_mpiwait_wtime, max_mpiwait_wtime, min_mpiwait_wtime,
             sum_num_messages, max_num_messages, min_num_messages);
   }
}

void DMEM_PrintParCSRMatrix(hypre_ParCSRMatrix *A, char *filename)
{
   int my_id, num_procs;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);
   HYPRE_Real *A_data;
   HYPRE_Int *A_i, *A_j;
   FILE *file_ptr;

   for (HYPRE_Int p = 0; p < num_procs; p++){
      if (my_id == p){
         if (p == 0){
            file_ptr = fopen(filename, "w");
         }
         else {
            file_ptr = fopen(filename, "a");
         }

         HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
         HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
         HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
         HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   
        // printf("%d: %d %d\n", my_id, first_row_index, first_col_diag);

         A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
         A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
         A_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A));
         for (HYPRE_Int i = 0; i < num_rows; i++){
            for (HYPRE_Int jj = A_i[i]; jj < A_i[i+1]; jj++){
               HYPRE_Int ii = A_j[jj];
               fprintf(file_ptr, "%d %d %.16e\n", first_row_index+i+1, first_col_diag+ii+1, A_data[jj]);
            }
         }
        

         A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A));
         A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A));
         A_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A));
         for (HYPRE_Int i = 0; i < num_rows; i++){
            for (HYPRE_Int jj = A_i[i]; jj < A_i[i+1]; jj++){
               HYPRE_Int ii = A_j[jj];
               fprintf(file_ptr, "%d %d %e\n", first_row_index+i+1, col_map_offd[ii]+1, A_data[jj]);
            }
         }

         fclose(file_ptr);
      }
      MPI_Barrier(comm);
   }
}


void DMEM_Gridk_PrintParCSRMatrix(hypre_ParCSRMatrix *A, char *filename)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   HYPRE_Real *A_data;
   HYPRE_Int *A_i, *A_j;
   FILE *file_ptr;

   for (HYPRE_Int p = 0; p < num_procs; p++){
      if (my_id == p){
         if (p == 0){
            file_ptr = fopen(filename, "w");
         }
         else {
            file_ptr = fopen(filename, "a");
         }

         HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
         HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
         HYPRE_Int num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
         HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

         A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
         A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
         A_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A));
         for (HYPRE_Int i = 0; i < num_rows; i++){
            for (HYPRE_Int jj = A_i[i]; jj < A_i[i+1]; jj++){
               HYPRE_Int ii = A_j[jj];
               fprintf(file_ptr, "%d %d %.16e\n", first_row_index+i+1, first_col_diag+ii+1, A_data[jj]);
            }
         }
        

         A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A));
         A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A));
         A_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A));
         for (HYPRE_Int i = 0; i < num_rows; i++){
            for (HYPRE_Int jj = A_i[i]; jj < A_i[i+1]; jj++){
               HYPRE_Int ii = A_j[jj];
               fprintf(file_ptr, "%d %d %e\n", first_row_index+i+1, col_map_offd[ii]+1, A_data[jj]);
            }
         }

         fclose(file_ptr);
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
}

HYPRE_Real InnerProd(hypre_Vector *x_local,
                     hypre_Vector *y_local,
                     MPI_Comm comm)
{
   HYPRE_Real inner_prod;
   HYPRE_Real local_inner_prod = hypre_SeqVectorInnerProd(x_local, y_local);

   hypre_MPI_Allreduce(&local_inner_prod,
                       &inner_prod,
                       1,
                       HYPRE_MPI_REAL,
                       hypre_MPI_SUM,
                       comm);
   return inner_prod;
}

HYPRE_Real InnerProdFlag(hypre_Vector *x_local,
                         hypre_Vector *y_local,
                         MPI_Comm comm,
                         HYPRE_Real my_flag,
                         HYPRE_Real *sum_flags)
{
   HYPRE_Real inner_prod;
   HYPRE_Real local_inner_prod = hypre_SeqVectorInnerProd(x_local, y_local);

   HYPRE_Real sendbuf[2] = {local_inner_prod, my_flag};
   HYPRE_Real recvbuf[2];
   hypre_MPI_Allreduce(sendbuf,
                       recvbuf,
                       2,
                       HYPRE_MPI_REAL,
                       hypre_MPI_SUM,
                       comm);
   *sum_flags = recvbuf[1];
   return recvbuf[0];
}

/* y = y + x ./ scale */
void DMEM_HypreParVector_Ivaxpy(hypre_ParVector *y, hypre_ParVector *x, HYPRE_Complex *scale, HYPRE_Int size)
{
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   HYPRE_Real *y_local_data = hypre_VectorData(y_local);
   HYPRE_Real *x_local_data = hypre_VectorData(x_local);

#if defined(HYPRE_USING_CUDA)
   hypre_SeqVectorPrefetch(y_local, HYPRE_MEMORY_DEVICE);
   hypre_SeqVectorPrefetch(x_local, HYPRE_MEMORY_DEVICE);
   hypreDevice_IVAXPY(size, scale, x_local_data, y_local_data);
#else
   for (int i = 0; i < size; i++){
      y_local_data[i] += x_local_data[i] / scale[i];
   }
#endif
}

/* y = y + alpha * x */
void DMEM_HypreParVector_Axpy(hypre_ParVector *y, hypre_ParVector *x, HYPRE_Complex alpha, HYPRE_Int size)
{
   hypre_ParVectorAxpy(alpha, x, y);
}

/* copy x into y */
void DMEM_HypreParVector_Copy(hypre_ParVector *y, hypre_ParVector *x, HYPRE_Int size)
{
   hypre_ParVectorCopy(x, y);
}

/* set y to alpha */
void DMEM_HypreParVector_Set(hypre_ParVector *y, HYPRE_Complex alpha, HYPRE_Int size)
{
   hypre_ParVectorSetConstantValues(y, alpha); 
}

/* y = alpha * y */
void DMEM_HypreParVector_Scale(hypre_ParVector *y, HYPRE_Complex alpha, HYPRE_Int size)
{
   hypre_ParVectorScale(alpha, y);
}

// TODO: turn the DMEM_HypreRealArray functions into hypre_SeqVector calls
void DMEM_HypreRealArray_Prefetch(HYPRE_Real *y, HYPRE_Int size, HYPRE_Int to_location)
{
   hypre_TMemcpy(y, y, HYPRE_Real, size, to_location, HYPRE_MEMORY_SHARED);   
}

void DMEM_HypreRealArray_Copy(HYPRE_Real *y, HYPRE_Real *x, HYPRE_Int size)
{
   if (vecop_machine == HYPRE_MEMORY_SHARED){
      #if defined(HYPRE_USING_CUDA)
         DMEM_HypreRealArray_Prefetch(x, size, HYPRE_MEMORY_DEVICE);
         DMEM_HypreRealArray_Prefetch(y, size, HYPRE_MEMORY_DEVICE);
         #if defined(HYPRE_USING_CUBLAS)
            HYPRE_CUBLAS_CALL( cublasDcopy(hypre_HandleCublasHandle(hypre_handle), size, x, 1, y, 1) );
         #else
            HYPRE_THRUST_CALL( copy_n, x, size, y );
         #endif
         hypre_SyncCudaComputeStream(hypre_handle);
      #else
         for (int i = 0; i < size; i++){
            y[i] = x[i];
         }
      #endif
   }
   else {
      for (int i = 0; i < size; i++){
         y[i] = x[i];
      }
   }
}

void DMEM_HypreRealArray_Axpy(HYPRE_Real *y, HYPRE_Real *x, HYPRE_Real alpha, HYPRE_Int size)
{
   if (vecop_machine == HYPRE_MEMORY_SHARED){
      #if defined(HYPRE_USING_CUDA)
         DMEM_HypreRealArray_Prefetch(x, size, HYPRE_MEMORY_DEVICE);
         DMEM_HypreRealArray_Prefetch(y, size, HYPRE_MEMORY_DEVICE);
         #if defined(HYPRE_USING_CUBLAS)
            HYPRE_CUBLAS_CALL( cublasDaxpy(hypre_HandleCublasHandle(hypre_handle), size, &alpha, x, 1, y, 1) );
         #else
            HYPRE_THRUST_CALL( transform, x, x + size, y, y, alpha * _1 + _2 );
         #endif
         hypre_SyncCudaComputeStream(hypre_handle);
      #else
         for (int i = 0; i < size; i++){
            y[i] += alpha * x[i];
         }  
      #endif
   }
   else {
      for (int i = 0; i < size; i++){
         y[i] += alpha * x[i];
      }
   }
}

void DMEM_HypreRealArray_Set(HYPRE_Real *y, HYPRE_Real alpha, HYPRE_Int size)
{
   if (vecop_machine == HYPRE_MEMORY_SHARED){
      #if defined(HYPRE_USING_CUDA)
         DMEM_HypreRealArray_Prefetch(y, size, HYPRE_MEMORY_DEVICE);
         #if defined(HYPRE_USING_CUBLAS)
            HYPRE_CUBLAS_CALL( cublasDscal(hypre_HandleCublasHandle(hypre_handle), size, &alpha, y_data, 1) );
         #else
            HYPRE_THRUST_CALL( transform, y, y + size, y, alpha * _1 );
         #endif
         hypre_SyncCudaComputeStream(hypre_handle);
      #else
         for (int i = 0; i < size; i++){
            y[i] = alpha;
         }
      #endif
   }
   else {
      for (int i = 0; i < size; i++){
         y[i] = alpha;
      }
   }
}

void DMEM_WriteCSR(CSR A, char *out_str, int base, OrderingData P, MPI_Comm comm)
{
   int num_procs, my_id;
   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);

   int row, col, k;
   double elem;
   FILE *out_file;
   if (my_id == 0) remove(out_str);

   for (int p = 0; p < num_procs; p++){
      if (p == my_id){
         out_file = fopen(out_str, "a");
         for (int i = 0; i < A.n; i++){
            for (int j = A.j_ptr[i]; j < A.j_ptr[i+1]; j++){
               row = P.disp[my_id] + i;
               col = A.i[j];
               elem = A.val[j];
               fprintf(out_file, "%d   %d   %e\n", row+base, col+base, elem);
            }
         }
         fclose(out_file);
      }
      MPI_Barrier(comm);
   }
}
