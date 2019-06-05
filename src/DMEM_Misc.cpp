#include "DMEM_Main.hpp"
#include "DMEM_Misc.hpp"


void DMEM_PrintOutput(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   double solve_wtime, residual_wtime, prolong_wtime, restrict_wtime, smooth_wtime, correct_wtime, comm_wtime, start_wtime, end_wtime;

   hypre_ParVector *r = dmem_all_data->vector_fine.r;
   dmem_all_data->output.r_norm2 = sqrt(hypre_ParVectorInnerProd(r, r));

   MPI_Reduce(&(dmem_all_data->output.solve_wtime),    &solve_wtime,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.residual_wtime), &residual_wtime,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.restrict_wtime), &restrict_wtime,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.prolong_wtime),  &prolong_wtime,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.smooth_wtime),   &smooth_wtime,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.correct_wtime),  &correct_wtime,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.comm_wtime),     &comm_wtime,      1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.start_wtime),    &start_wtime,     1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&(dmem_all_data->output.end_wtime),      &end_wtime,       1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   solve_wtime /= (double)num_procs;
   residual_wtime /= (double)num_procs;
   restrict_wtime /= (double)num_procs;
   prolong_wtime /= (double)num_procs;
   smooth_wtime /= (double)num_procs;
   correct_wtime /= (double)num_procs;
   comm_wtime /= (double)num_procs;
   start_wtime /= (double)num_procs;
   end_wtime /= (double)num_procs;

   if (my_id == 0){
      char print_str[1000];
      if (dmem_all_data->input.format_output_flag == 0){
         strcpy(print_str, "Setup stats:\n"
                           "Solve stats:\n"
                           "\tRelative Residual 2-norm = %e\n"
                           "\tSolve time = %e, breakdown:\n"
                           "\t\tResidual time = %e\n"
                           "\t\tProlong time = %e\n"
                           "\t\tRestrict time = %e\n"
                           "\t\tSmooth time = %e\n"
                           "\t\tCorrect time = %e\n"
                           "\t\tComm time = %e\n"
                           "\t\tStart time = %e\n"
                           "\t\tEnd time = %e\n");
      }
      else {
         strcpy(print_str, "%e ");
      }
      printf(print_str,
             dmem_all_data->output.r_norm2/dmem_all_data->output.r0_norm2,
             solve_wtime,
             residual_wtime,
             prolong_wtime,
             restrict_wtime,
             smooth_wtime,
             correct_wtime,
             comm_wtime,
             start_wtime,
             end_wtime);
   }
}

void DMEM_PrintParCSRMatrix(hypre_ParCSRMatrix *A, char *filename)
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
      MPI_Barrier(MPI_COMM_WORLD);
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

   MPI_Allreduce(&local_inner_prod,
                 &inner_prod,
                 1,
                 HYPRE_MPI_REAL,
                 hypre_MPI_SUM,
                 comm);
   return inner_prod;
}
