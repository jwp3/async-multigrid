#include "DMEM_Main.hpp"
#include "DMEM_Misc.hpp"


void DMEM_PrintOutput(DMEM_AllData *dmem_all_data)
{
   int my_id, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int mean_smooth_sweeps;
   int mean_cycles;
   double mean_smooth_wtime;
   double mean_restrict_wtime;
   double mean_residual_wtime;
   double mean_prolong_wtime;
   double mean_correct;

   hypre_ParVector *r = dmem_all_data->vector_fine.r;
   dmem_all_data->output.r_norm2 =
      sqrt(hypre_ParVectorInnerProd(r, r));

   if (my_id == 0){
      char print_str[1000];
      if (dmem_all_data->input.format_output_flag == 0){
         strcpy(print_str, "Setup stats:\n"
                           "Solve stats:\n"
                           "\tRelative Residual 2-norm = %e\n");
      }
      else {
         strcpy(print_str, "%e ");
      }
      printf(print_str,
             dmem_all_data->output.r_norm2/dmem_all_data->output.r0_norm2);
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
