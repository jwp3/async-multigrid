#include "Main.hpp"
#include "SMEM_MatVec.hpp"
#include "SMEM_Smooth.hpp"
#include "Misc.hpp"

void SMEM_Sync_Parfor_Vcycle(AllData *all_data)
{
   #pragma omp parallel
   {
      int fine_grid, coarse_grid;
      for (int level = 0; level < all_data->grid.num_levels-1; level++){
         fine_grid = level;
         coarse_grid = level + 1;
         SMEM_Sync_Parfor_Jacobi(all_data,
                                 all_data->matrix.A[fine_grid],
                                 all_data->vector.f[fine_grid],
                                 all_data->vector.u[fine_grid],
                                 all_data->vector.u_prev[fine_grid],
                                 all_data->input.num_pre_smooth_sweeps);
         SMEM_Sync_Parfor_Residual(all_data,
                                   all_data->matrix.A[fine_grid],
                                   all_data->vector.f[fine_grid],
                                   all_data->vector.u[fine_grid],
                                   all_data->vector.y[fine_grid],
                                   all_data->vector.r[fine_grid]);
         SMEM_Sync_Parfor_MatVec(all_data,
                                 all_data->matrix.R[fine_grid],
                                 all_data->vector.r[fine_grid],
                                 all_data->vector.f[coarse_grid]);
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
            all_data->vector.u[coarse_grid][i] = 0;
         }
      }
   }

   int this_grid = all_data->grid.num_levels-1;
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
           all_data->vector.f[this_grid],
           all_data->vector.u[this_grid],
           &(all_data->pardiso.info.error));

   #pragma omp parallel
   {
      int fine_grid, coarse_grid;
      for (int level = all_data->grid.num_levels-2; level > -1; level--){
         fine_grid = level;
         coarse_grid = level + 1;
         SMEM_Sync_Parfor_MatVec(all_data,
                                 all_data->matrix.P[fine_grid],
                                 all_data->vector.u[coarse_grid],
                                 all_data->vector.e[fine_grid]);
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
            all_data->vector.u[fine_grid][i] += all_data->vector.e[fine_grid][i];
         }
         SMEM_Sync_Parfor_Jacobi(all_data,
                    all_data->matrix.A[fine_grid],
                    all_data->vector.f[fine_grid],
                    all_data->vector.u[fine_grid],
                    all_data->vector.u_prev[fine_grid],
                    all_data->input.num_post_smooth_sweeps);
      }
   }
}

void SMEM_Sync_Parfor_AFACx_Vcycle(AllData *all_data)
{
   #pragma omp parallel
   {
      int fine_grid, coarse_grid;
      for (int level = 0; level < all_data->grid.num_levels-1; level++){
         fine_grid = level;
         coarse_grid = level + 1;
         SMEM_Sync_Parfor_MatVec(all_data,
                    all_data->matrix.R[fine_grid],
                    all_data->vector.r[fine_grid],
                    all_data->vector.r[coarse_grid]);
      }
   }

   int this_grid = all_data->grid.num_levels-1;
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
           all_data->vector.r[this_grid],
           all_data->vector.u_fine[this_grid],
           &(all_data->pardiso.info.error));

   #pragma omp parallel
   {
      int fine_grid, coarse_grid, this_grid;
      for (int level = all_data->grid.num_levels-1; level > -1; level--){
         if (level != all_data->grid.num_levels-1){
            fine_grid = level;
            coarse_grid = level + 1;

            #pragma omp for
            for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
               all_data->vector.u_fine[fine_grid][i] = 0;
            }
            #pragma omp for
            for (int i = 0; i < all_data->grid.n[coarse_grid]; i++){
               all_data->vector.u_coarse[coarse_grid][i] = 0;
            }

            SMEM_Sync_Parfor_Jacobi(all_data,
                                    all_data->matrix.A[coarse_grid],
                                    all_data->vector.r[coarse_grid],
                                    all_data->vector.u_coarse[coarse_grid],
                                    all_data->vector.u_coarse_prev[coarse_grid],
                                    all_data->input.num_coarse_smooth_sweeps);
            SMEM_Sync_Parfor_MatVec(all_data,
                                    all_data->matrix.P[fine_grid],
                                    all_data->vector.u_coarse[coarse_grid],
                                    all_data->vector.e[fine_grid]);
            SMEM_Sync_Parfor_Residual(all_data,
                                      all_data->matrix.A[fine_grid],
                                      all_data->vector.r[fine_grid],
                                      all_data->vector.e[fine_grid],
                                      all_data->vector.y[fine_grid],
                                      all_data->vector.r_fine[fine_grid]);
            SMEM_Sync_Parfor_Jacobi(all_data,
                                    all_data->matrix.A[fine_grid],
                                    all_data->vector.r_fine[fine_grid],
                                    all_data->vector.u_fine[fine_grid],
                                    all_data->vector.u_fine_prev[fine_grid],
                                    all_data->input.num_fine_smooth_sweeps);

         }


         this_grid = level;
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[this_grid]; i++){
            all_data->vector.e[this_grid][i] = all_data->vector.u_fine[this_grid][i];
         }
         if (level > 0){
            for (int inner_level = level; inner_level > 0; inner_level--){
               fine_grid = inner_level - 1;
               coarse_grid = inner_level;
               SMEM_Sync_Parfor_MatVec(all_data,
                          all_data->matrix.P[fine_grid],
                          all_data->vector.e[coarse_grid],
                          all_data->vector.e[fine_grid]);
            }
         }
         fine_grid = 0;
         #pragma omp for
         for (int i = 0; i < all_data->grid.n[fine_grid]; i++){
            all_data->vector.u[fine_grid][i] += all_data->vector.e[fine_grid][i];
         }
      }
   }
}
