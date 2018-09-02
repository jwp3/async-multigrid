#include "Main.hpp"
#include "SEQ_Smooth.hpp"
#include "SEQ_MatVec.hpp"
#include "Misc.hpp"

void SEQ_Vcycle(AllData *all_data)
{
   int fine_grid, coarse_grid, this_grid;
   int n;

   fine_grid = 0;
   SEQ_Residual(all_data,
                all_data->matrix.A_array[fine_grid],
                all_data->vector.f_array[fine_grid],
                all_data->vector.u_array[fine_grid],
                all_data->vector.y_array[fine_grid],
                all_data->vector.r_array[fine_grid]);
   all_data->output.r0_norm2 = 
      Norm2(all_data->vector.r_array[fine_grid], all_data->grid.n[fine_grid]);

   for (int k = 0; k < all_data->input.num_cycles; k++){
      for (int level = 0; level < all_data->grid.num_levels-1; level++){
         fine_grid = level;
         coarse_grid = level + 1;
         if (all_data->input.smoother == JACOBI){
            SEQ_Jacobi(all_data,
                       all_data->matrix.A_array[fine_grid],
                       all_data->vector.f_array[fine_grid],
                       all_data->vector.u_array[fine_grid],
           	    all_data->vector.u_prev_array[fine_grid],
                       all_data->input.num_pre_smooth_sweeps);
         }
         else {
            SEQ_GaussSeidel(all_data,
                            all_data->matrix.A_array[fine_grid],
                            all_data->vector.f_array[fine_grid],
                            all_data->vector.u_array[fine_grid],
                            all_data->input.num_pre_smooth_sweeps);
         }
         SEQ_Residual(all_data,
                      all_data->matrix.A_array[fine_grid],
                      all_data->vector.f_array[fine_grid],
                      all_data->vector.u_array[fine_grid],
                      all_data->vector.y_array[fine_grid],
                      all_data->vector.r_array[fine_grid]);
         SEQ_MatVecT(all_data,
                     all_data->matrix.R_array[fine_grid],
                     all_data->vector.r_array[fine_grid],
                     all_data->vector.f_array[coarse_grid]);
         n = all_data->grid.n[coarse_grid];
         for (int i = 0; i < n; i++){
            all_data->vector.u_array[coarse_grid][i] = 0;
         }
      }

      this_grid = all_data->grid.num_levels-1;
      PARDISO(all_data->pardiso.info.pt,
              &(all_data->pardiso.info.maxfct),
              &(all_data->pardiso.info.mnum),
              &(all_data->pardiso.info.mtype),
              &(all_data->pardiso.info.phase),
              &(all_data->pardiso.csr.n),
              all_data->pardiso.csr.a,
              all_data->pardiso.csr.ia,
              all_data->pardiso.csr.ja,
              &(all_data->pardiso.info.idum),
              &(all_data->pardiso.info.nrhs),
              all_data->pardiso.info.iparm,
              &(all_data->pardiso.info.msglvl),
              all_data->vector.f_array[this_grid],
              all_data->vector.u_array[this_grid],
              &(all_data->pardiso.info.error));


      for (int level = all_data->grid.num_levels-2; level > -1; level--){
         fine_grid = level;
         coarse_grid = level + 1;
         SEQ_MatVec(all_data,
                    all_data->matrix.P_array[fine_grid],
                    all_data->vector.u_array[coarse_grid],
                    all_data->vector.e_array[fine_grid]);
         n = all_data->grid.n[fine_grid];
         for (int i = 0; i < n; i++){
            all_data->vector.u_array[fine_grid][i] += all_data->vector.e_array[fine_grid][i];
         }
         if (all_data->input.smoother == JACOBI){
            SEQ_Jacobi(all_data,
                       all_data->matrix.A_array[fine_grid],
                       all_data->vector.f_array[fine_grid],
                       all_data->vector.u_array[fine_grid],
                       all_data->vector.u_prev_array[fine_grid],
                       all_data->input.num_post_smooth_sweeps);
         }
         else {
            SEQ_GaussSeidel(all_data,
                            all_data->matrix.A_array[fine_grid],
                            all_data->vector.f_array[fine_grid],
                            all_data->vector.u_array[fine_grid],
                            all_data->input.num_post_smooth_sweeps);
         }
      }
      fine_grid = 0;
      SEQ_Residual(all_data,
                   all_data->matrix.A_array[fine_grid],
                   all_data->vector.f_array[fine_grid],
                   all_data->vector.u_array[fine_grid],
                   all_data->vector.y_array[fine_grid],
                   all_data->vector.r_array[fine_grid]);
      all_data->output.r_norm2 =
         Norm2(all_data->vector.r_array[fine_grid], all_data->grid.n[fine_grid]);
      if (all_data->input.print_reshist_flag == 1){
         printf("%d\t%e\n",
                k, all_data->output.r_norm2/all_data->output.r0_norm2);
      }
   }
}
