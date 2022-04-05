#include "Main.hpp"
#include "Misc.hpp"

#ifdef USE_MFEM
//#include "../mfem_quartz/mfem-4.0/linalg/solvers.hpp"

void MFEM_Elasticity(AllData *all_data,
                     HYPRE_IJMatrix *Aij)
{
   bool static_cond = false;
   int flux_averaging = 0;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(all_data->mfem.mesh_file);
   int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < all_data->mfem.ref_levels; l++){
      mesh.UniformRefinement();
   }

   if (mesh.NURBSext){
      if (all_data->input.test_problem == MFEM_ELAST_AMR){
         mesh.SetCurvature(2);
      }
      else {
         mesh.DegreeElevate(all_data->mfem.order, all_data->mfem.order);
      }
   }

   mesh.EnsureNCMesh();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
  // FiniteElementCollection *fec = new H1_FECollection(all_data->mfem.order, dim);
  //  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   H1_FECollection fec(all_data->mfem.order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec, dim, Ordering::byVDIM);

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
      Vector pull_force(pmesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm b(&fespace);
   if (all_data->input.test_problem == MFEM_ELAST_AMR){
      b.AddDomainIntegrator(new VectorBoundaryLFIntegrator(f));
   }
   else {
      b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   }

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of ,
   //    which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.
   Vector lambda(pmesh.attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh.attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   ParBilinearForm a(&fespace);
   BilinearFormIntegrator *integ = new ElasticityIntegrator(lambda_func, mu_func);
   a.AddDomainIntegrator(integ);

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }

   const int tdim = dim*(dim+1)/2;
   L2_FECollection flux_fec(all_data->mfem.order, dim);
   ParFiniteElementSpace flux_fespace(&pmesh, &flux_fec, tdim);
   ParFiniteElementSpace smooth_flux_fespace(&pmesh, &fec, tdim);
   L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace, smooth_flux_fespace);
   //ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace);
   //estimator.SetFluxAveraging(flux_averaging);

   // 11. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   HypreParMatrix A_out;
   Vector B_out, X_out;

   while(1){
      HypreParMatrix A;
      Vector B, X;

      a.Assemble();
      b.Assemble();

      Array<int> ess_tdof_list, ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[0] = 1;
      //if (all_data->input.test_problem == MFEM_ELAST_AMR){
      //   Vector zero_vec(dim);
      //   zero_vec = 0.0;
      //   ConstantCoefficient zero_coeff(0.0);
      //   VectorConstantCoefficient zero_vec_coeff(zero_vec);
      //   x.ProjectBdrCoefficient(zero_coeff, ess_bdr);
      //}
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 15. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
      int cdofs = fespace.GlobalTrueVSize();
      if (all_data->input.test_problem == MFEM_ELAST ||
          cdofs >= all_data->mfem.max_amr_dofs){
         A_out = A;
         B_out = B;
         X_out = X;
         break;
      }

      // 16. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //     solve the linear system with PCG.
      //GSSmoother M(A);
      //PCG(A, M, B, X, 3, 2000, 1e-12, 0.0);
      //CG(A, B, X, 3, 2000, 1e-12, 0.0);
      HypreBoomerAMG *amg = new HypreBoomerAMG(A);
      //if (amg_elast && !a->StaticCondensationIsEnabled())
      //{
      //   amg->SetElasticityOptions(fespace);
      //}
      //else
      //{
         amg->SetSystemsOptions(dim);
      //}
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-8);
      pcg->SetMaxIter(500);
      pcg->SetPrintLevel(3);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);

      // 17. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      a.RecoverFEMSolution(X, b, x);

      // 19. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh); 
      exit(0);
      if (refiner.Stop()){
         A_out = A;
         B_out = B;
         X_out = X;
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
      a.Update();
      b.Update();
   }

  // if (all_data->input.mfem_test_error_flag == 1){
  //    for (int i = 0; i < A.Height(); i++){
  //       B[i] = 1.0;
  //    }

  //   // GSSmoother M(A);
  //   // PCG(A, M, B, X, all_data->input.mfem_solve_print_flag, 200, 1e-12, 0.0);     

  //    all_data->mfem.u = (double *)malloc(A.Height() * sizeof(double));
  //    for (int i = 0; i < A.Height(); i++){
  //       all_data->mfem.u[i] = X[i];
  //    }
  // }

   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, A_out.Height()-1, 0, A_out.Height()-1, Aij);
   HYPRE_IJMatrixSetObjectType(*Aij, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(*Aij);

   all_data->hypre.b_values = (HYPRE_Real *)malloc(A_out.Height() * sizeof(HYPRE_Real));

   SparseMatrix A_diag;
   A_out.GetDiag(A_diag);
   for (int i = 0; i < A_out.Height(); i++){
      all_data->hypre.b_values[i] = B_out[i];

      Array<int> mfem_cols;
      Vector mfem_srow;
      A_diag.GetRow(i, mfem_cols, mfem_srow);

      int nnz = mfem_srow.Size();

      double *values = (double *)malloc(nnz * sizeof(double));
      int *cols = (int *)malloc(nnz * sizeof(int));
      
      int k = 0;
      for (int j = 0; j < nnz; j++){
         cols[k] = mfem_cols[j];
         values[k] = mfem_srow[j];
         k++;
      }

      ///* Set the values for row i */
      HYPRE_IJMatrixSetValues(*Aij, 1, &nnz, &i, cols, values);
   }

   //hypre_ParCSRMatrix *A_hypre = A_out.GetHypreMatrix();
   //char buffer[100];
   //sprintf(buffer, "A.txt");
   //PrintCSRMatrix(hypre_ParCSRMatrixDiag(A_hypre), buffer, 0);

   // 14. Free the used memory.
  // delete a;
  // delete b;
  // delete fespace;
  // delete fec;
  // delete mesh;
}

#endif
