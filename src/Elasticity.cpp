#include "Main.hpp"

#ifdef USE_MFEM
void MFEM_Elasticity(AllData *all_data,
                     HYPRE_IJMatrix *Aij)
{
   bool static_cond = false;
   int flux_averaging = 0;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(all_data->mfem.mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext){
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
  // FiniteElementCollection *fec = new H1_FECollection(all_data->mfem.order, dim);
  //  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   H1_FECollection fec(all_data->mfem.order, dim);
   FiniteElementSpace fespace(mesh, &fec, dim);

      // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;

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

   LinearForm *b = new LinearForm(&fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   Vector zero_vec(dim);
   zero_vec = 0.0;
   VectorConstantCoefficient zero_vec_coeff(zero_vec);
   GridFunction x(&fespace);
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

   BilinearForm *a = new BilinearForm(&fespace);
   BilinearFormIntegrator *integ = new ElasticityIntegrator(lambda_func, mu_func);
   a->AddDomainIntegrator(integ);

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }

   const int tdim = dim*(dim+1)/2;
   FiniteElementSpace flux_fespace(mesh, &fec, tdim);
   ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace);
   estimator.SetFluxAveraging(flux_averaging);

   // 11. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   SparseMatrix A;
   Vector B, X;

   while(1){
      a->Assemble();
      b->Assemble();

      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(zero_vec_coeff, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 15. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      const int copy_interior = 1;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B, copy_interior);
      
      int cdofs = fespace.GetTrueVSize();
     // if (cdofs >= all_data->mfem.max_amr_dofs){
         break;
     // }

      // 16. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //     solve the linear system with PCG.
     // TODO: fix PCG
     // GSSmoother M(A);
     // PCG(A, M, B, X, 3, 2000, 1e-12, 0.0);

      // 17. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      a->RecoverFEMSolution(X, *b, x);

      // 19. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(*mesh); 
      if (refiner.Stop()){
         break;
      }

      // 20. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations later
      //     since we'll have a good initial guess of x in the next step.
      //     Internally, FiniteElementSpace::Update() calculates an
      //     interpolation matrix which is then used by GridFunction::Update().
      fespace.Update();
      x.Update();

      // 21. Inform also the bilinear and linear forms that the space has
      //     changed.
      a->Update();
      b->Update();
   }

   if (all_data->input.mfem_test_error_flag == 1){
      for (int i = 0; i < A.Height(); i++){
         B[i] = 1.0;
      }

      // TODO: fix PCG
     // GSSmoother M(A);
     // PCG(A, M, B, X, all_data->input.mfem_solve_print_flag, 200, 1e-12, 0.0);     

      all_data->mfem.u = (double *)malloc(A.Height() * sizeof(double));
      for (int i = 0; i < A.Height(); i++){
         all_data->mfem.u[i] = X[i];
      }
   }

   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, A.Height()-1, 0, A.Height()-1, Aij);
   HYPRE_IJMatrixSetObjectType(*Aij, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(*Aij);

   for (int i = 0; i < A.Height(); i++){
     // Array<int> mfem_cols;
     // Vector mfem_srow;
     // A.GetRow(i, mfem_cols, mfem_srow);

     // int nnz = mfem_srow.Size();

     // double *values = (double *)malloc(nnz * sizeof(double));
     // int *cols = (int *)malloc(nnz * sizeof(int));

     // for (int j = 0; j < nnz; j++){
     //    cols[j] = mfem_cols[j];
     //    values[j] = mfem_srow[j];
     // }

      int nnz = A.RowSize(i);
      double *values = A.GetRowEntries(i);
      int *cols = A.GetRowColumns(i);

      /* Set the values for row i */
      HYPRE_IJMatrixSetValues(*Aij, 1, &nnz, &i, cols, values);
   }

   // 14. Free the used memory.
  // delete a;
  // delete b;
  // delete fespace;
  // delete fec;
  // delete mesh;
}

//                                MFEM Example 17
//
// Compile with: make ex17
//
// Sample runs:
//
//       ex17 -m ../data/beam-tri.mesh
//       ex17 -m ../data/beam-quad.mesh
//       ex17 -m ../data/beam-tet.mesh
//       ex17 -m ../data/beam-hex.mesh
//       ex17 -m ../data/beam-quad.mesh -r 2 -o 3
//       ex17 -m ../data/beam-quad.mesh -r 2 -o 2 -a 1 -k 1
//       ex17 -m ../data/beam-hex.mesh -r 2 -o 2
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam using symmetric or
//               non-symmetric discontinuous Galerkin (DG) formulation.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               Dirichlet, u=u_D on the fixed part of the boundary, namely
//               boundary attributes 1 and 2; on the rest of the boundary we use
//               sigma(u).n=0 b.c. The geometry of the domain is assumed to be
//               as follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (fixed, nonzero)
//
//               The example demonstrates the use of high-order DG vector finite
//               element spaces with the linear DG elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and function vector-coefficient objects. The use of
//               non-homogeneous Dirichlet b.c. imposed weakly, is also
//               illustrated.
//
//               We recommend viewing examples 2 and 14 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Initial displacement, used for Dirichlet boundary conditions on boundary
// attributes 1 and 2.
void InitDisplacement(const Vector &x, Vector &u);

// A Coefficient for computing the components of the stress.
class StressCoefficient : public Coefficient
{
protected:
   Coefficient &lambda, &mu;
   GridFunction *u; // displacement
   int si, sj; // component of the stress to evaluate, 0 <= si,sj < dim

   DenseMatrix grad; // auxiliary matrix, used in Eval

public:
   StressCoefficient(Coefficient &lambda_, Coefficient &mu_)
      : lambda(lambda_), mu(mu_), u(NULL), si(0), sj(0) { }

   void SetDisplacement(GridFunction &u_) { u = &u_; }
   void SetComponent(int i, int j) { si = i; sj = j; }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};


void MFEM_Elasticity2(AllData *all_data,
                      HYPRE_IJMatrix *Aij)
{
   double alpha = -1.0;
   double kappa = -1.0;

   if (kappa < 0)
   {
      kappa = (all_data->mfem.order+1)*(all_data->mfem.order+1);
   }

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(all_data->mfem.mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < all_data->mfem.ref_levels; l++)
   {
      mesh.UniformRefinement();
   }
   // Since NURBS meshes do not support DG integrators, we convert them to
   // regular polynomial mesh of the specified (solution) order.
   if (mesh.NURBSext) { mesh.SetCurvature(all_data->mfem.order); }

   // 4. Define a DG vector finite element space on the mesh. Here, we use
   //    Gauss-Lobatto nodal basis because it gives rise to a sparser matrix
   //    compared to the default Gauss-Legendre nodal basis.
   DG_FECollection fec(all_data->mfem.order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec, dim);

   // 5. In this example, the Dirichlet boundary conditions are defined by
   //    marking boundary attributes 1 and 2 in the marker Array 'dir_bdr'.
   //    These b.c. are imposed weakly, by adding the appropriate boundary
   //    integrators over the marked 'dir_bdr' to the bilinear and linear forms.
   //    With this DG formulation, there are no essential boundary conditions.
   Array<int> ess_tdof_list; // no essential b.c. (empty list)
   Array<int> dir_bdr(mesh.bdr_attributes.Max());
   dir_bdr = 0;
   dir_bdr[0] = 1; // boundary attribute 1 is Dirichlet
   dir_bdr[1] = 1; // boundary attribute 2 is Dirichlet

   // 6. Define the DG solution vector 'x' as a finite element grid function
   //    corresponding to fespace. Initialize 'x' using the 'InitDisplacement'
   //    function.
   GridFunction x(&fespace);
   VectorFunctionCoefficient init_x(dim, InitDisplacement);
   x.ProjectCoefficient(init_x);

   // 7. Set up the Lame constants for the two materials. They are defined as
   //    piece-wise (with respect to the element attributes) constant
   //    coefficients, i.e. type PWConstCoefficient.
   Vector lambda(mesh.attributes.Max());
   lambda = 1.0;      // Set lambda = 1 for all element attributes.
   lambda(0) = 50.0;  // Set lambda = 50 for element attribute 1.
   PWConstCoefficient lambda_c(lambda);
   Vector mu(mesh.attributes.Max());
   mu = 1.0;      // Set mu = 1 for all element attributes.
   mu(0) = 50.0;  // Set mu = 50 for element attribute 1.
   PWConstCoefficient mu_c(mu);

   // 8. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this example, the linear form b(.) consists
   //    only of the terms responsible for imposing weakly the Dirichlet
   //    boundary conditions, over the attributes marked in 'dir_bdr'. The
   //    values for the Dirichlet boundary condition are taken from the
   //    VectorFunctionCoefficient 'x_init' which in turn is based on the
   //    function 'InitDisplacement'.
   LinearForm b(&fespace);
   b.AddBdrFaceIntegrator(
      new DGElasticityDirichletLFIntegrator(
         init_x, lambda_c, mu_c, alpha, kappa), dir_bdr);
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) on the DG finite element space
   //    corresponding to the linear elasticity integrator with coefficients
   //    lambda and mu as defined above. The additional interior face integrator
   //    ensures the weak continuity of the displacement field. The additional
   //    boundary face integrator works together with the boundary integrator
   //    added to the linear form b(.) to impose weakly the Dirichlet boundary
   //    conditions.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new ElasticityIntegrator(lambda_c, mu_c));
   a.AddInteriorFaceIntegrator(
      new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa));
   a.AddBdrFaceIntegrator(
      new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa), dir_bdr);

   a.Assemble();

   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   
   if (all_data->input.mfem_test_error_flag == 1){
      for (int i = 0; i < A.Height(); i++){
         B[i] = 1.0;
      }

      // TODO: fix PCG
     // GSSmoother M(A);
     // PCG(A, M, B, X, all_data->input.mfem_solve_print_flag, 200, 1e-12, 0.0);

      all_data->mfem.u = (double *)malloc(A.Height() * sizeof(double));
      for (int i = 0; i < A.Height(); i++){
         all_data->mfem.u[i] = X[i];
      }
   }

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
}


void InitDisplacement(const Vector &x, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*x(0);
}


double StressCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "displacement field is not set");

   double L = lambda.Eval(T, ip);
   double M = mu.Eval(T, ip);
   u->GetVectorGradient(T, grad);
   if (si == sj)
   {
      double div_u = grad.Trace();
      return L*div_u + 2*M*grad(si,si);
   }
   else
   {
      return M*(grad(si,sj) + grad(sj,si));
   }
}

#endif
