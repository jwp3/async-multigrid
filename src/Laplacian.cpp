#include "Main.hpp"

void Laplacian_2D_5pt(HYPRE_IJMatrix *A,
                      int n,
		      int N,
                      int ilower,
                      int iupper)
{
   /* go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
   */

   int nnz;
   double values[5];
   int cols[5];

   for (int i = ilower; i <= iupper; i++)
   {
      nnz = 0;

      /* The left identity block:position i-n */
      if ((i-n)>=0)
      {
         cols[nnz] = i-n;
         values[nnz] = -1.0;
         nnz++;
      }

      /* The left -1: position i-1 */
      if (i%n)
      {
         cols[nnz] = i-1;
         values[nnz] = -1.0;
         nnz++;
      }

      /* Set the diagonal: position i */
      cols[nnz] = i;
      values[nnz] = 4.0;
      nnz++;

      /* The right -1: position i+1 */
      if ((i+1)%n)
      {
         cols[nnz] = i+1;
         values[nnz] = -1.0;
         nnz++;
      }

      /* The right identity block:position i+n */
      if ((i+n)< N)
      {
         cols[nnz] = i+n;
         values[nnz] = -1.0;
         nnz++;
      }

      /* Set the values for row i */
      HYPRE_IJMatrixSetValues(*A, 1, &nnz, &i, cols, values);
   }
}

