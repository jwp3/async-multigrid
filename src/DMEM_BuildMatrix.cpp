#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "Misc.hpp"
#include "_hypre_utilities.h"

using namespace std;

static inline HYPRE_Int sign_double(HYPRE_Real a)
{
   return ( (0.0 < a) - (0.0 > a) );
}

void DMEM_BuildHypreMatrix(DMEM_AllData *dmem_all_data,
                           HYPRE_ParCSRMatrix *A_ptr,
                           HYPRE_ParVector *rhs_ptr,
                           MPI_Comm comm,
                           HYPRE_Int nx, HYPRE_Int ny, HYPRE_Int nz,
                           HYPRE_Real cx, HYPRE_Real cy, HYPRE_Real cz,
                           HYPRE_Real ax, HYPRE_Real ay, HYPRE_Real az,
                           HYPRE_Real eps,
                           int atype)
{
   HYPRE_Int P, Q, R;

   HYPRE_ParCSRMatrix A;
   HYPRE_ParVector rhs;

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

   if (dmem_all_data->input.test_problem == VARDIFCONV_3D7PT){
      A = (HYPRE_ParCSRMatrix) GenerateVarDifConv(comm, nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);
      *rhs_ptr = rhs;
   }
   if (dmem_all_data->input.test_problem == DIFCONV_3D7PT){
      HYPRE_Real hinx, hiny, hinz;
      HYPRE_Int sign_prod;

      hinx = 1./(nx+1);
      hiny = 1./(ny+1);
      hinz = 1./(nz+1);

      values = hypre_CTAlloc(HYPRE_Real,  7, HYPRE_MEMORY_HOST);

      values[0] = 0.;

      if (0 == atype) /* forward scheme for conv */
      {
         values[1] = -cx/(hinx*hinx);
         values[2] = -cy/(hiny*hiny);
         values[3] = -cz/(hinz*hinz);
         values[4] = -cx/(hinx*hinx) + ax/hinx;
         values[5] = -cy/(hiny*hiny) + ay/hiny;
         values[6] = -cz/(hinz*hinz) + az/hinz;

         if (nx > 1)
         {
            values[0] += 2.0*cx/(hinx*hinx) - 1.*ax/hinx;
         }
         if (ny > 1)
         {
            values[0] += 2.0*cy/(hiny*hiny) - 1.*ay/hiny;
         }
         if (nz > 1)
         {
            values[0] += 2.0*cz/(hinz*hinz) - 1.*az/hinz;
         }
      }
      else if (1 == atype) /* backward scheme for conv */
      {
         values[1] = -cx/(hinx*hinx) - ax/hinx;
         values[2] = -cy/(hiny*hiny) - ay/hiny;
         values[3] = -cz/(hinz*hinz) - az/hinz;
         values[4] = -cx/(hinx*hinx);
         values[5] = -cy/(hiny*hiny);
         values[6] = -cz/(hinz*hinz);

         if (nx > 1)
         {
            values[0] += 2.0*cx/(hinx*hinx) + 1.*ax/hinx;
         }
         if (ny > 1)
         {
            values[0] += 2.0*cy/(hiny*hiny) + 1.*ay/hiny;
         }
         if (nz > 1)
         {
            values[0] += 2.0*cz/(hinz*hinz) + 1.*az/hinz;
         }
      }
      else if (3 == atype) /* upwind scheme */
      {
         sign_prod = sign_double(cx) * sign_double(ax);
         if (sign_prod == 1) /* same sign use back scheme */
         {
            values[1] = -cx/(hinx*hinx) - ax/hinx;
            values[4] = -cx/(hinx*hinx);
            if (nx > 1)
            {
               values[0] += 2.0*cx/(hinx*hinx) + 1.*ax/hinx;
            }
         }
         else /* diff sign use forward scheme */
         {
            values[1] = -cx/(hinx*hinx);
            values[4] = -cx/(hinx*hinx) + ax/hinx;
            if (nx > 1)
            {
               values[0] += 2.0*cx/(hinx*hinx) - 1.*ax/hinx;
            }
         }

         sign_prod = sign_double(cy) * sign_double(ay);
         if (sign_prod == 1) /* same sign use back scheme */
         {
            values[2] = -cy/(hiny*hiny) - ay/hiny;
            values[5] = -cy/(hiny*hiny);
            if (ny > 1)
            {
               values[0] += 2.0*cy/(hiny*hiny) + 1.*ay/hiny;
            }
         }
         else /* diff sign use forward scheme */
         {
            values[2] = -cy/(hiny*hiny);
            values[5] = -cy/(hiny*hiny) + ay/hiny;
            if (ny > 1)
            {
               values[0] += 2.0*cy/(hiny*hiny) - 1.*ay/hiny;
            }
         }

         sign_prod = sign_double(cz) * sign_double(az);
         if (sign_prod == 1) /* same sign use back scheme */
         {
            values[3] = -cz/(hinz*hinz) - az/hinz;
            values[6] = -cz/(hinz*hinz);
            if (nz > 1)
            {
               values[0] += 2.0*cz/(hinz*hinz) + 1.*az/hinz;
            }
         }
         else /* diff sign use forward scheme */
         {
            values[3] = -cz/(hinz*hinz);
            values[6] = -cz/(hinz*hinz) + az/hinz;
            if (nz > 1)
            {
               values[0] += 2.0*cz/(hinz*hinz) - 1.*az/hinz;
            }
         }
      }
      else /* centered difference scheme */
      {
         values[1] = -cx/(hinx*hinx) - ax/(2.*hinx);
         values[2] = -cy/(hiny*hiny) - ay/(2.*hiny);
         values[3] = -cz/(hinz*hinz) - az/(2.*hinz);
         values[4] = -cx/(hinx*hinx) + ax/(2.*hinx);
         values[5] = -cy/(hiny*hiny) + ay/(2.*hiny);
         values[6] = -cz/(hinz*hinz) + az/(2.*hinz);

         if (nx > 1)
         {
            values[0] += 2.0*cx/(hinx*hinx);
         }
         if (ny > 1)
         {
            values[0] += 2.0*cy/(hiny*hiny);
         }
         if (nz > 1)
         {
            values[0] += 2.0*cz/(hinz*hinz);
         }
      }

      A = (HYPRE_ParCSRMatrix) GenerateDifConv(hypre_MPI_COMM_WORLD,
                                               nx, ny, nz, P, Q, R, p, q, r, values);
      hypre_TFree(values, HYPRE_MEMORY_HOST); 
   }
   else {
      values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

      values[0] = 26.0;
      if (nx == 1 || ny == 1 || nz == 1)
         values[0] = 8.0;
      if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
         values[0] = 2.0;
      values[1] = -1.;

      A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(comm, nx, ny, nz, P, Q, R, p, q, r, values);

      hypre_TFree(values, HYPRE_MEMORY_HOST);
   }

   *A_ptr = A;
}
