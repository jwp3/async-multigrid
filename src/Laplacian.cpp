#include "Main.hpp"

void Laplacian_2D_5pt(HYPRE_IJMatrix *A, int n)
{
   /* go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
   */

   int N = n*n;
   
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, N-1, 0, N-1, A);
   HYPRE_IJMatrixSetObjectType(*A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(*A);

   int nnz;
   double values[5];
   int cols[5];

   for (int i = 0; i < N; i++)
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

void Laplacian_3D_7pt(HYPRE_ParCSRMatrix *A_ptr,
                      int n)
{
   HYPRE_Int nx, ny, nz;
   HYPRE_Int P, Q, R;

   HYPRE_ParCSRMatrix A;

   HYPRE_Int num_procs, myid;
   HYPRE_Int p, q, r;
   HYPRE_Real cx, cy, cz;
   HYPRE_Real *values;


   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   nx = n;
   ny = n;
   nz = n;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   values = hypre_CTAlloc(HYPRE_Real,  4, HYPRE_MEMORY_HOST);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD,
                                              nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;
}

void Laplacian_3D_27pt(HYPRE_ParCSRMatrix *A_ptr,
                       int n)
{
   HYPRE_Int nx, ny, nz;
   HYPRE_Int P, Q, R;

   HYPRE_ParCSRMatrix A;

   HYPRE_Int num_procs, myid;
   HYPRE_Int p, q, r;
   HYPRE_Real *values;


   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   nx = n;
   ny = n;
   nz = n;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );


   values = hypre_CTAlloc(HYPRE_Real,  2, HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values, HYPRE_MEMORY_HOST);

   *A_ptr = A;
}

// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

void MFEM_Laplacian(AllData *all_data,
                    HYPRE_IJMatrix *Aij)
{
   bool static_cond = false;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(all_data->mfem.mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < all_data->mfem.ref_levels; l++){
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(all_data->mfem.order, 1));
   }
  // mesh->SetCurvature(2);

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   fec = new H1_FECollection(all_data->mfem.order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
  // cout << "Number of finite element unknowns: "
  //      << fespace->GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector X, B;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   if (all_data->input.mfem_test_error_flag == 1){
      for (int i = 0; i < A.Height(); i++){
         B[i] = 1.0;   
      }

      GSSmoother M(A);
      PCG(A, M, B, X, all_data->input.mfem_solve_print_flag, 200, 1e-12, 0.0);
    
      all_data->mfem.u = (double *)malloc(A.Height() * sizeof(double)); 
      for (int i = 0; i < A.Height(); i++){
         all_data->mfem.u[i] = X[i];
      }
   }

   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, A.Height()-1, 0, A.Height()-1, Aij);
   HYPRE_IJMatrixSetObjectType(*Aij, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(*Aij);

   for (int i = 0; i < A.Height(); i++){

      Array<int> mfem_cols;
      Vector mfem_srow;
      A.GetRow(i, mfem_cols, mfem_srow);

      int nnz = mfem_srow.Size();

      double *values = (double *)malloc(nnz * sizeof(double));
      int *cols = (int *)malloc(nnz * sizeof(int));
      
      for (int j = 0; j < nnz; j++){
         cols[j] = mfem_cols[j];
         values[j] = mfem_srow[j];
      }

      /* Set the values for row i */
      HYPRE_IJMatrixSetValues(*Aij, 1, &nnz, &i, cols, values);
   }

  // cout << "Size of linear system: " << A.Height() << endl;

   // 14. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;
}
