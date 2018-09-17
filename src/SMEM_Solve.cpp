#include "Main.hpp"
#include "Misc.hpp"
#include "SEQ_MatVec.hpp"
#include "SEQ_AMG.hpp"
#include "SMEM_MatVec.hpp"
#include "SMEM_Sync_AMG.hpp"
#include "SMEM_Async_AMG.hpp"
#include "SMEM_Smooth.hpp"
#include "SEQ_Smooth.hpp"

void SMEM_Solve(AllData *all_data)
{
   int fine_grid = 0;
   SEQ_Residual(all_data,
                all_data->matrix.A[fine_grid],
                all_data->vector.f[fine_grid],
                all_data->vector.u[fine_grid],
                all_data->vector.y[fine_grid],
                all_data->vector.r[fine_grid]);
   all_data->output.r0_norm2 =
      Norm2(all_data->vector.r[fine_grid], all_data->grid.n[fine_grid]);

   double start = omp_get_wtime();
   if (all_data->input.async_flag == 1){
      SMEM_Async_Add_AMG(all_data);
      all_data->output.solve_wtime = omp_get_wtime() - start;
      if (all_data->input.num_threads == 1){
         SEQ_Residual(all_data,
                      all_data->matrix.A[fine_grid],
                      all_data->vector.f[fine_grid],
                      all_data->vector.u[fine_grid],
                      all_data->vector.y[fine_grid],
                      all_data->vector.r[fine_grid]);
      }
      else{
         #pragma omp parallel
         {
            SMEM_Sync_Parfor_Residual(all_data,
                                      all_data->matrix.A[fine_grid],
                                      all_data->vector.f[fine_grid],
                                      all_data->vector.u[fine_grid],
                                      all_data->vector.y[fine_grid],
                                      all_data->vector.r[fine_grid]);
         }
      }
      all_data->output.r_norm2 =
         Parfor_Norm2(all_data->vector.r[fine_grid], all_data->grid.n[fine_grid]);
   }
   else{
      if (all_data->input.print_reshist_flag == 1 &&
          all_data->input.format_output_flag == 0){
         printf("\nIters\tRel. Res. 2-norm\n"
                "-----\t----------------\n");
      }
      if (all_data->input.print_reshist_flag == 1 &&
          all_data->input.format_output_flag == 0){
         printf("%d\t%e\n", 0, 1.);
      }
      for (int k = 0; k < all_data->input.num_cycles; k++){
         if (all_data->input.solver == MULTADD){
            if (all_data->input.thread_part_type == ALL_LEVELS){
               SMEM_Sync_Add_Vcycle(all_data);
            }
            else{
            } 
         }
         else if (all_data->input.solver == AFACX){
            if (all_data->input.thread_part_type == ALL_LEVELS){
               SMEM_Sync_Add_Vcycle(all_data);
            }
            else{
               if (all_data->input.num_threads == 1){
                  SEQ_AFACx_Vcycle(all_data);
               }
               else{
                  SMEM_Sync_Parfor_AFACx_Vcycle(all_data);
               }
            }
         }
         else{
            if (all_data->input.num_threads == 1){
               SEQ_Vcycle(all_data);
            }
            else{
               SMEM_Sync_Parfor_Vcycle(all_data);
            }
         }
         all_data->output.solve_wtime = omp_get_wtime() - start;
         if (all_data->input.num_threads == 1){
            SEQ_Residual(all_data,
                         all_data->matrix.A[fine_grid],
                         all_data->vector.f[fine_grid],
                         all_data->vector.u[fine_grid],
                         all_data->vector.y[fine_grid],
                         all_data->vector.r[fine_grid]);
         }
         else{
            #pragma omp parallel
            {
               SMEM_Sync_Parfor_Residual(all_data,
                                         all_data->matrix.A[fine_grid],
                                         all_data->vector.f[fine_grid],
                                         all_data->vector.u[fine_grid],
                                         all_data->vector.y[fine_grid],
                                         all_data->vector.r[fine_grid]);
            }
         }
         all_data->output.r_norm2 =
            Parfor_Norm2(all_data->vector.r[fine_grid], all_data->grid.n[fine_grid]);
         if (all_data->input.print_reshist_flag == 1 &&
             all_data->input.format_output_flag == 0){
            printf("%d\t%e\n",
                   k+1, all_data->output.r_norm2/all_data->output.r0_norm2);
         }
      }
   }
}

void SMEM_Smooth(AllData *all_data,
                 hypre_CSRMatrix *A,
                 HYPRE_Real *f,
                 HYPRE_Real *u,
                 HYPRE_Real *y,
                 HYPRE_Real *r,
                 int num_sweeps,
                 int level,
                 int ns, int ne)
{
   if (all_data->input.num_threads > 1){
      if (all_data->input.thread_part_type == ALL_LEVELS){
         if (all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL){
            SMEM_Sync_HybridJacobiGaussSeidel(all_data, A, f, u, y, num_sweeps, level, ns, ne);
         }
         else if (all_data->input.smoother == SYMM_JACOBI){
            SMEM_Sync_SymmetricJacobi(all_data, A, f, u, y, r, num_sweeps, level, ns, ne);
         }
         else if (all_data->input.smoother == ASYNC_GAUSS_SEIDEL){
            SMEM_Async_GaussSeidel(all_data, A, f, u, num_sweeps, level, ns, ne);
         }
         else if (all_data->input.smoother == SEMI_ASYNC_GAUSS_SEIDEL){
            SMEM_SemiAsync_GaussSeidel(all_data, A, f, u, num_sweeps, level, ns, ne);
         }
         else {
            SMEM_Sync_Jacobi(all_data, A, f, u, y, num_sweeps, level, ns, ne);
         }
      }
      else{
         if (all_data->input.smoother == HYBRID_JACOBI_GAUSS_SEIDEL){
            SMEM_Sync_Parfor_HybridJacobiGaussSeidel(all_data, A, f, u, y, num_sweeps, level);
         }
         else if (all_data->input.smoother == ASYNC_GAUSS_SEIDEL){
            SMEM_Async_Parfor_GaussSeidel(all_data, A, f, u, num_sweeps, level);
         }
         else if (all_data->input.smoother == SEMI_ASYNC_GAUSS_SEIDEL){
            SMEM_SemiAsync_Parfor_GaussSeidel(all_data, A, f, u, num_sweeps, level);
         }
         else {
            SMEM_Sync_Parfor_Jacobi(all_data, A, f, u, y, num_sweeps, level);
         }
      }
   }
   else{
      if (all_data->input.smoother == GAUSS_SEIDEL){
         SEQ_GaussSeidel(all_data, A, f, u, num_sweeps);
      }
      else {
         SEQ_Jacobi(all_data, A, f, u, y, num_sweeps);
      }
   }
}
