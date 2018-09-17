#include "Main.hpp"

// Sample runs:  ex2 -m ../data/beam-tri.mesh
//               ex2 -m ../data/beam-quad.mesh
//               ex2 -m ../data/beam-tet.mesh
//               ex2 -m ../data/beam-hex.mesh
//               ex2 -m ../data/beam-quad.mesh -o 3 -sc
//               ex2 -m ../data/beam-quad-nurbs.mesh
//               ex2 -m ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)
//
//               The example demonstrates the use of high-order and NURBS vector
//               finite element spaces with the linear elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and vector coefficient objects. Static condensation is
//               also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

void MFEM_Elasticity(AllData *all_data,
                     HYPRE_IJMatrix *Aij)
{
   bool static_cond = false;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(all_data->mfem.mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(all_data->mfem.order, all_data->mfem.order);
   }

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < all_data->mfem.ref_levels; l++){
      mesh->UniformRefinement();
   }
  // mesh->SetCurvature(2);

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   fec = new H1_FECollection(all_data->mfem.order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

      // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "pull down" force on the Neumann part
   //    of the boundary and phi_i are the basis functions in the finite element
   //    fespace. The force is defined by the VectorArrayCoefficient object f,
   //    which is a vector of Coefficient objects. The fact that f is non-zero
   //    on boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++){
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.
   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, A.Height()-1, 0, A.Height()-1, Aij);
   HYPRE_IJMatrixSetObjectType(*Aij, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(*Aij);

   for (int i = 0; i < A.Height(); i++)
   {

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

   // 14. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;
}
