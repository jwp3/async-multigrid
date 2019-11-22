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
      values = hypre_CTAlloc(HYPRE_Real,  2, dmem_all_data->input.hypre_memory);

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
                          hypre_ParCSRMatrix **A,
                          MPI_Comm comm)
{
   int num_procs, my_id;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   bool static_cond = false;
   HYPRE_IJMatrix A_ij;

   Mesh *mesh = new Mesh(dmem_all_data->mfem.mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   if (mesh->NURBSext){
      mesh->DegreeElevate(dmem_all_data->mfem.order, dmem_all_data->mfem.order);
   }

   for (int l = 0; l < dmem_all_data->mfem.ref_levels; l++){
      mesh->UniformRefinement();
   }
   mesh->EnsureNCMesh();

   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;
   for (int l = 0; l < dmem_all_data->mfem.par_ref_levels; l++){
      pmesh->UniformRefinement();
   }
  // pmesh->EnsureNCMesh();


   FiniteElementCollection *fec;
   fec = new H1_FECollection(dmem_all_data->mfem.order, dim);
   ParFiniteElementSpace *fespace;

   if (pmesh->NURBSext) {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
   }
   else {
      fec = new H1_FECollection(dmem_all_data->mfem.order, dim);
      fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
   }

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

   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b->Assemble();

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
   a->Assemble();

   HypreParMatrix A_mfem;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A_mfem, X, B);

   *A = A_mfem.GetHypreMatrix();

  // HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, A_mfem.Height()-1, 0, A_mfem.Height()-1, A_ij);
  // HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR);
  // HYPRE_IJMatrixInitialize(A_ij);

  // for (int i = 0; i < A_mfem.Height(); i++){

  //    Array<int> mfem_cols;
  //    Vector mfem_srow;
  //    A_mfem.GetRow(i, mfem_cols, mfem_srow);

  //    int nnz = mfem_srow.Size();

  //    double *values = (double *)malloc(nnz * sizeof(double));
  //    int *cols = (int *)malloc(nnz * sizeof(int));

  //    for (int j = 0; j < nnz; j++){
  //       cols[j] = mfem_cols[j];
  //       values[j] = mfem_srow[j];
  //    }

  //    HYPRE_IJMatrixSetValues(A_ij, 1, &nnz, &i, cols, values);
  // }

  // HYPRE_IJMatrixAssemble(A_ij);
  // HYPRE_IJMatrixGetObject(A_ij, (void**)A);

  // delete a;
   delete b;
  // delete fespace;
   delete fec;
   delete pmesh;
}
