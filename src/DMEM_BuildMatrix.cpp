#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "Misc.hpp"
#include "_hypre_utilities.h"

using namespace std;
using namespace mfem;

static inline HYPRE_Int sign_double(HYPRE_Real a)
{
   return ((0.0 < a) - (0.0 > a));
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

      values = hypre_CTAlloc(HYPRE_Real,  7, dmem_all_data->input.hypre_memory);

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
      hypre_TFree(values, dmem_all_data->input.hypre_memory); 
   }
   else if (dmem_all_data->input.test_problem == LAPLACE_3D7PT){
      values = hypre_CTAlloc(HYPRE_Real,  4, dmem_all_data->input.hypre_memory);

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

      hypre_TFree(values, dmem_all_data->input.hypre_memory);
   }
   else {
      values = hypre_CTAlloc(HYPRE_Real, 2, dmem_all_data->input.hypre_memory);

      values[0] = 26.0;
      if (nx == 1 || ny == 1 || nz == 1)
         values[0] = 8.0;
      if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
         values[0] = 2.0;
      values[1] = -1.;

      A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(comm, nx, ny, nz, P, Q, R, p, q, r, values);

      hypre_TFree(values, dmem_all_data->input.hypre_memory);
   }

   *A_ptr = A;
}

void DMEM_BuildMfemMatrix(DMEM_AllData *dmem_all_data,
                          hypre_ParCSRMatrix **A_ptr,
                          hypre_ParVector **b_ptr,
                          MPI_Comm comm)
{
   int num_procs, my_id;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   bool static_cond = false;
   bool amg_elast = 0;
   HYPRE_IJMatrix A_ij;
   Mesh *mesh = new Mesh("./mfem_quartz/mfem-4.0/data/beam-hex.mesh", 1, 1);
  // Mesh *mesh = new Mesh(dmem_all_data->mfem.mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

  // if (dmem_all_data->input.test_problem == MFEM_ELAST_AMR){

  // }
  // else {
      dmem_all_data->hypre.num_functions = dim;
  // }

   if (mesh->NURBSext){
      mesh->DegreeElevate(dmem_all_data->mfem.order, dmem_all_data->mfem.order);
   }

   for (int l = 0; l < dmem_all_data->mfem.ref_levels; l++){
      mesh->UniformRefinement();
   }
   if (dmem_all_data->input.test_problem == MFEM_ELAST_AMR){
      if (mesh->NURBSext){
         mesh->SetCurvature(2);
      }
      mesh->EnsureNCMesh();
   }

   ParMesh *pmesh = new ParMesh(comm, *mesh);
   if (dmem_all_data->input.test_problem == MFEM_ELAST_AMR){
      mesh->Clear();
   }
   else {
      delete mesh;
   }
   for (int l = 0; l < dmem_all_data->mfem.par_ref_levels; l++){
      pmesh->UniformRefinement();
   }
  //pmesh->EnsureNCMesh();

   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   const bool use_nodal_fespace = pmesh->NURBSext && !amg_elast;
   if (dmem_all_data->input.test_problem == MFEM_ELAST_AMR){
      fec = new H1_FECollection(dmem_all_data->mfem.order, dim);
      fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
   }
   else {
      if (use_nodal_fespace){
         fec = NULL;
         fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
      }
      else{
         fec = new H1_FECollection(dmem_all_data->mfem.order, dim);
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
      }
   }

   HYPRE_Int size = fespace->GlobalTrueVSize();

   Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++){
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(pmesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   ParGridFunction x(fespace);
   x = 0.0;

   Vector lambda(pmesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   BilinearFormIntegrator *integ = new ElasticityIntegrator(lambda_func, mu_func);
   a->AddDomainIntegrator(integ);

   if (static_cond) { a->EnableStaticCondensation(); }

   HypreParMatrix A;
   Vector B, X;

   if (dmem_all_data->input.test_problem == MFEM_ELAST_AMR){
      Vector zero_vec(dim);
      zero_vec = 0.0;
      VectorConstantCoefficient zero_vec_coeff(zero_vec);

      const int tdim = dim*(dim+1)/2;
      L2_FECollection flux_fec(dmem_all_data->mfem.order, dim);
      ParFiniteElementSpace flux_fespace(pmesh, &flux_fec, tdim);
      ParFiniteElementSpace smooth_flux_fespace(pmesh, fec, tdim);
      L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace, smooth_flux_fespace);

      ThresholdRefiner refiner(estimator);
      refiner.SetTotalErrorFraction(0.7);

      const int max_dofs = 50000;
      const int max_amr_itr = 20;
      for (int it = 0; it <= max_amr_itr; it++)
      {
         HYPRE_Int global_dofs = fespace->GlobalTrueVSize();

         a->Assemble();
         b->Assemble();

         Array<int> ess_tdof_list;
         x.ProjectBdrCoefficient(zero_vec_coeff, ess_bdr);
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        // HypreParMatrix A;
        // Vector B, X;
         const int copy_interior = 1;
         a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B, copy_interior);

         HypreBoomerAMG *amg = new HypreBoomerAMG(A);
         amg->SetSystemsOptions(dim);
        // amg->SetPrintLevel(2);
        // amg->Mult(B, X);
        // delete amg;
         CGSolver pcg(A.GetComm());
         pcg.SetPreconditioner(*amg);
         pcg.SetOperator(A);
         pcg.SetRelTol(1e-6);
         pcg.SetMaxIter(500);
         pcg.SetPrintLevel(0); // set to 3 to print the first and the last iterations only
         pcg.Mult(B, X);

         a->RecoverFEMSolution(X, *b, x);

         if (global_dofs > max_dofs)
         {
            break;
         }

         refiner.Apply(*pmesh);
         if (refiner.Stop())
         {
            break;
         }

         fespace->Update();
         x.Update();

         if (pmesh->Nonconforming())
         {
            pmesh->Rebalance();

            fespace->Update();
            x.Update();
         }

         a->Update();
         b->Update();
      }
   }
   else {
      a->Assemble();
      b->Assemble();
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   }

   *A_ptr = A.GetHypreMatrix();
   if (dmem_all_data->input.rhs_type == RHS_FROM_PROBLEM){
      HypreParVector *B_par = b->ParallelAssemble();
      *b_ptr = B_par->GetHypreVector();
   }

  // HypreBoomerAMG *amg = new HypreBoomerAMG(A);
  // if (amg_elast && !a->StaticCondensationIsEnabled())
  // {
  //    amg->SetElasticityOptions(fespace);
  // }
  // else
  // {
  //    amg->SetSystemsOptions(dim);
  // }
  // amg->SetPrintLevel(3);
  // amg->Mult(B, X);






  // HyprePCG *pcg = new HyprePCG(A);
  // pcg->SetTol(1e-8);
  // pcg->SetMaxIter(500);
  // pcg->SetPrintLevel(2);
  // pcg->SetPreconditioner(*amg);
  // pcg->Mult(B, X);

  // HYPRE_IJMatrix Aij;
  // HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, A.GetGlobalNumRows()-1, 0, A.GetGlobalNumRows()-1, &Aij);
  // HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);
  // HYPRE_IJMatrixInitialize(Aij);

  // SparseMatrix diag, offd;
  // HYPRE_Int *cmap;
  // A.GetDiag(diag);
  // A.GetOffd(offd, cmap);

  // for (int i = 0; i < diag.Height(); i++){

  //    Array<int> mfem_diag_cols, mfem_offd_cols;
  //    Vector mfem_diag_srow, mfem_offd_srow;
  //    diag.GetRow(i, mfem_diag_cols, mfem_diag_srow);
  //    offd.GetRow(i, mfem_offd_cols, mfem_offd_srow);

  //    int nnz_diag = mfem_diag_srow.Size();
  //    int nnz_offd = mfem_offd_srow.Size();
  //    int nnz = nnz_diag + nnz_offd;

  //    double *values = (double *)malloc(nnz * sizeof(double));
  //    int *cols = (int *)malloc(nnz * sizeof(int));

  //    int jj = 0;
  //    for (int j = 0; j < nnz_diag; j++){
  //       cols[jj] = mfem_diag_cols[j];
  //       values[jj] = mfem_diag_srow[j];
  //       jj++;
  //    }
  //    for (int j = 0; j < nnz_offd; j++){
  //       cols[jj] = mfem_offd_cols[j];
  //       values[jj] = mfem_offd_srow[j];
  //       jj++;
  //    }

  //    /* Set the values for row i */
  //    HYPRE_IJMatrixSetValues(Aij, 1, &nnz, &i, cols, values);
  // }

  // HYPRE_IJMatrixAssemble(Aij);
  // void *object;
  // HYPRE_IJMatrixGetObject(Aij, &object);
  // *A_ptr = (hypre_ParCSRMatrix *)object;


   //delete pcg;
   //delete amg;
   //delete a;
   //delete b;
   //if (fec)
   //{
   //   delete fespace;
   //   delete fec;
   //}
   //delete pmesh; 
}
