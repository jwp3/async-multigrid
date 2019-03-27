#include "Main.hpp"
#include "DMEM_Main.hpp"

void DMEM_Laplacian_3D_27pt(DMEM_AllData *dmem_all_data,
			    HYPRE_ParCSRMatrix *A_ptr,
			    MPI_Comm comm,
			    HYPRE_Int nx,
			    HYPRE_Int ny,
			    HYPRE_Int nz)
{
   HYPRE_Int P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int num_procs, my_id;
   HYPRE_Int p, q, r;
   HYPRE_Real *values;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if (P*Q*R != num_procs ||
       P > nx ||
       Q > ny ||
       R > nz)
   {
      if (my_id == 0)
         hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and my_id */
   p = my_id % P;
   q = ((my_id - p)/P) % Q;
   r = (my_id - p - P*q)/(P*Q);

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(comm, nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;
}
