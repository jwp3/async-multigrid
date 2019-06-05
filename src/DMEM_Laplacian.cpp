#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "Misc.hpp"

using namespace std;

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

   vector<int> divs = Divisors(num_procs);

   if (divs.size() == 2){
      P = 1;
      Q = num_procs;
      R = 1;      
   }
   else {
      HYPRE_Int x, y, z;
      int break_flag = 0;
      x = y = z = 1;
      for (int i = 0; i < divs.size(); i++){
         for (int j = 0; j < divs.size(); j++){
            for (int k = 0; k < divs.size(); k++){
               if (divs[k] > nz || break_flag == 1){
                  break;
               }
               x = divs[i]; y = divs[j]; z = divs[k];
               if (x*y*z == num_procs){
                  break_flag = 1;
               }
            }
            if (divs[j] > ny || break_flag == 1){
               break;
            }
         }
         if (divs[i] > nx || break_flag == 1){
            break;
         }
      }
      P = x;
      Q = y;
      R = z;
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if (P*Q*R != num_procs || P > nx || Q > ny || R > nz){
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
