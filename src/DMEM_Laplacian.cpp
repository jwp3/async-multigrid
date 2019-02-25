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

   HYPRE_Int num_procs, myid;
   HYPRE_Int p, q, r;
   HYPRE_Real *values;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

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
      if (myid == 0)
         hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = ((myid - p)/P) % Q;
   r = (myid - p - P*q)/(P*Q);

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(comm, nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;
}
