#include "Main.hpp"
#include "Misc.hpp"
#include "SEQ_MatVec.hpp"
#include "SEQ_AMG.hpp"
#include "SMEM_Sync_AMG.hpp"

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
   if (all_data->input.print_reshist_flag == 1 &&
       all_data->input.format_output_flag == 0){
      printf("\nIters\tRel. Res. 2-norm\n"
             "-----\t----------------\n");
   }
   if (all_data->input.print_reshist_flag == 1){
      printf("%d\t%e\n", 0, 1.);
   }
   for (int k = 0; k < all_data->input.num_cycles; k++){
      if (all_data->input.solver == MULT_ADD){
     
      }
      else if (all_data->input.solver == AFACX){
         if (all_data->input.thread_part_type == ALL_LEVELS){
            SMEM_Sync_AFACx_Vcycle(all_data);
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
      SEQ_Residual(all_data,
                   all_data->matrix.A[fine_grid],
                   all_data->vector.f[fine_grid],
                   all_data->vector.u[fine_grid],
                   all_data->vector.y[fine_grid],
                   all_data->vector.r[fine_grid]);
      all_data->output.r_norm2 =
         Norm2(all_data->vector.r[fine_grid], all_data->grid.n[fine_grid]);
      if (all_data->input.print_reshist_flag == 1){
         printf("%d\t%e\n",
                k+1, all_data->output.r_norm2/all_data->output.r0_norm2);
      }
   }
}