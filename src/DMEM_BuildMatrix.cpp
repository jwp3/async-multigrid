#include "Main.hpp"
#include "DMEM_Main.hpp"
#include "Misc.hpp"
#include "DMEM_Misc.hpp"
#include "DMEM_BuildMatrix.hpp"
#include "_hypre_utilities.h"

using namespace std;
using namespace mfem;

void List_to_Metis(MetisGraph *G,
                   Triplet *T,
                   std::vector<std::list<int>> col_list,
                   std::vector<std::list<double>> elem_list);
void ReorderTriplet(Triplet T, CSR *A, OrderingData *P);
void ReadBinary_fread_metis(FILE *mat_file,
                            MetisGraph *G,
                            Triplet *T,
                            int symm_flag);
void Reorder(OrderingData *P, Triplet *T, CSR *A);
void CSRtoParHypreCSRMatrix(CSR B,
                            hypre_ParCSRMatrix **A_ptr,
                            OrderingData P,
                            MPI_Comm comm);

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
//#if defined(HYPRE_USING_CUDA)
//   const char *device_config = "cuda";
//   Device device(device_config);
//#else
//   const char *device_config = "cpu";
//   Device device(device_config);
//#endif

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
   }
   mesh->EnsureNCMesh();

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

      for (int it = 0;; it++){
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
         return;

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

         if ((global_dofs > dmem_all_data->mfem.max_amr_dofs) || (it == dmem_all_data->mfem.max_amr_iters)){
            break;
         }

         refiner.Apply(*pmesh);
         if (refiner.Stop()){
            break;
         }

         fespace->Update();
         x.Update();

         if (pmesh->Nonconforming()){
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

void DMEM_DistributeHypreParCSRMatrix_FineToGridk(DMEM_AllData *dmem_all_data,
                                                  hypre_ParCSRMatrix *A,
                                                  hypre_ParCSRMatrix **B)
{
   int num_procs, my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   MPI_Comm my_comm = dmem_all_data->grid.my_comm;

   HYPRE_Int loc_num_procs, loc_my_id;
   MPI_Comm_rank(my_comm, &loc_my_id);
   MPI_Comm_size(my_comm, &loc_num_procs);

   HYPRE_Int my_grid = dmem_all_data->grid.my_grid;
   HYPRE_Int num_levels = dmem_all_data->grid.num_levels;
   
   HYPRE_Int num_procs_level, rest, **ps, **pe;
   ps = (HYPRE_Int **)malloc(num_levels * sizeof(HYPRE_Int *));
   pe = (HYPRE_Int **)malloc(num_levels * sizeof(HYPRE_Int *));
   //printf("%d, %d, %d\n", num_procs, num_my_procs, rest);

   int finest_level;
   if (dmem_all_data->input.res_compute_type == GLOBAL_RES){
      finest_level = dmem_all_data->input.coarsest_mult_level + 1;
   }
   else{
      finest_level = dmem_all_data->input.coarsest_mult_level;
   }
   for (int level = finest_level; level < num_levels; level++){
      ps[level] = (HYPRE_Int *)calloc(dmem_all_data->grid.num_procs_level[level], sizeof(HYPRE_Int));
      pe[level] = (HYPRE_Int *)calloc(dmem_all_data->grid.num_procs_level[level], sizeof(HYPRE_Int));

      num_procs_level = num_procs/dmem_all_data->grid.num_procs_level[level];
      rest = num_procs - num_procs_level*dmem_all_data->grid.num_procs_level[level];
      for (int p = 0; p < dmem_all_data->grid.num_procs_level[level]; p++){
         if (p < rest){
            ps[level][p] = p*num_procs_level + p;
            pe[level][p] = (p + 1)*num_procs_level + p + 1;
         }
         else {
            ps[level][p] = p*num_procs_level + rest;
            pe[level][p] = (p + 1)*num_procs_level + rest;
         }
      }
   }

  // printf("(%d,%d,%d): %d, %d\n", loc_my_id, my_grid, num_my_procs, ps, pe)

   double *recvbuf_v, *sendbuf_v;
   int *recvbuf_i, *sendbuf_i;
   int *recvbuf_j, *sendbuf_j;
   int *recvbuf, *sendbuf;
   int *rdispls, *recvcounts;
   int *sdispls, *sendcounts;
   int sendcount, recvcount;

   recvcounts = (int *)calloc(num_procs, sizeof(int));
   rdispls = (int *)calloc(num_procs, sizeof(int));
   sendcounts = (int *)calloc(num_procs, sizeof(int));
   sdispls = (int *)calloc(num_procs, sizeof(int));

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Int *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Real *diag_data = hypre_CSRMatrixData(diag);

   HYPRE_Int *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Real *offd_data = hypre_CSRMatrixData(offd);

   /* indicate to other processes that I need their diag and offd */
   recvbuf = (int *)calloc(num_procs, sizeof(int));
   sendbuf = (int *)calloc(num_procs, sizeof(int));

   HYPRE_Int *recv_flags = (HYPRE_Int *)calloc(num_procs, sizeof(HYPRE_Int));
   for (HYPRE_Int p = 0; p < num_procs; p++){
      if (p >= ps[my_grid][loc_my_id] && p < pe[my_grid][loc_my_id]){
         sendbuf[p] = 1;
         recv_flags[p] = 1;
      }
   }
  
   MPI_Alltoall(sendbuf,
                1,
                MPI_INT,
                recvbuf,
                1,
                MPI_INT,
                MPI_COMM_WORLD);

   /* send nnz */
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(diag) + hypre_CSRMatrixNumNonzeros(offd);
   int *send_flags = (int *)calloc(num_procs, sizeof(int));
   sendcount = 0;
   for (int p = 0; p < num_procs; p++){
      sendbuf[p] = 0;
      if (recvbuf[p] == 1){
         send_flags[p] = 1;
         sendbuf[p] = nnz;
      }
      /* we need sdispls and sendcount later when we send I,J,V */
      if (p > 0){
         sdispls[p] = sdispls[p-1] + sendbuf[p-1];
      }
      sendcounts[p] = sendbuf[p];
      sendcount += sendbuf[p];
   }

   MPI_Alltoall(sendbuf,
                1,
                MPI_INT,
                recvbuf,
                1,
                MPI_INT,
                MPI_COMM_WORLD);

 
   /* send I,J,V */
   recvcount = 0;
   for (HYPRE_Int p = 0; p < num_procs; p++){
      if (p > 0){
         rdispls[p] = rdispls[p-1] + recvbuf[p-1];
      }
      recvcounts[p] = recvbuf[p];
      recvcount += recvbuf[p];
   }

   free(sendbuf);
   free(recvbuf);

   recvbuf_i = (int *)calloc(recvcount, sizeof(int));
   recvbuf_j = (int *)calloc(recvcount, sizeof(int));
   recvbuf_v = (double *)calloc(recvcount, sizeof(double));

   sendbuf_j = (int *)calloc(sendcount, sizeof(int));
   sendbuf_i = (int *)calloc(sendcount, sizeof(int));
   sendbuf_v = (double *)calloc(sendcount, sizeof(double));

  // HYPRE_Int *I = (HYPRE_Int *)calloc(recvcount, sizeof(HYPRE_Int));
  // HYPRE_Int *J = (HYPRE_Int *)calloc(recvcount, sizeof(HYPRE_Int));
  // HYPRE_Real *V = (HYPRE_Real *)calloc(recvcount, sizeof(HYPRE_Real));

 
   HYPRE_Int k = 0;
   for (HYPRE_Int p = 0; p < num_procs; p++){
      if (send_flags[p] == 1){

         HYPRE_Int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
         HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
         HYPRE_Int num_rows = hypre_CSRMatrixNumRows(diag);
         HYPRE_Int *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

         for (HYPRE_Int i = 0; i < num_rows; i++){
            for (HYPRE_Int jj = diag_i[i]; jj < diag_i[i+1]; jj++){
               HYPRE_Int ii = diag_j[jj];

               sendbuf_i[k] = first_row_index+i;
               sendbuf_j[k] = first_col_diag+ii;
               sendbuf_v[k] = diag_data[jj];

               k++;
            }
         }

         for (HYPRE_Int i = 0; i < num_rows; i++){
            for (HYPRE_Int jj = offd_i[i]; jj < offd_i[i+1]; jj++){
               HYPRE_Int ii = offd_j[jj];

               sendbuf_i[k] = first_row_index+i;
               sendbuf_j[k] = col_map_offd[ii];
               sendbuf_v[k] = offd_data[jj];

               k++;
            }
         }
      }
   }
 
   MPI_Alltoallv(sendbuf_i,
                 sendcounts,
                 sdispls,
                 MPI_INT,
                 recvbuf_i,
                 recvcounts,
                 rdispls,
                 MPI_INT,
                 MPI_COMM_WORLD);

   MPI_Alltoallv(sendbuf_j,
                 sendcounts,
                 sdispls,
                 MPI_INT,
                 recvbuf_j,
                 recvcounts,
                 rdispls,
                 MPI_INT,
                 MPI_COMM_WORLD);

   MPI_Alltoallv(sendbuf_v,
                 sendcounts,
                 sdispls,
                 MPI_DOUBLE,
                 recvbuf_v,
                 recvcounts,
                 rdispls,
                 MPI_DOUBLE,
                 MPI_COMM_WORLD);

   
   HYPRE_IJMatrix ij_matrix;
   int ilower = MinInt(recvbuf_i, recvcount);
   int iupper = MaxInt(recvbuf_i, recvcount);
   int jlower = MinInt(recvbuf_j, recvcount);
   int jupper = MaxInt(recvbuf_j, recvcount);

   recvbuf = (int *)calloc(4*num_procs, sizeof(int));
   sendbuf = (int *)calloc(4*num_procs, sizeof(int));

   for (int p = 0; p < num_procs; p++){
      if (send_flags[p] == 1){
         int first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
         int last_row_index = hypre_ParCSRMatrixLastRowIndex(A);
         int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
         int last_col_diag = hypre_ParCSRMatrixLastColDiag(A);

         sendbuf[4*p] =   first_row_index;
         sendbuf[4*p+1] = last_row_index;
         sendbuf[4*p+2] = first_col_diag;
         sendbuf[4*p+3] = last_col_diag;
      }
   }
   
  // printf("%d: interp %d %d\n", my_id, iupper, jupper);

   MPI_Alltoall(sendbuf,
                4,
                MPI_INT,
                recvbuf,
                4,
                MPI_INT,
                MPI_COMM_WORLD);

   ilower = jlower = INT_MAX;//max(hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixGlobalNumCols(A))+1;
   iupper = jupper = INT_MIN;
   for (int p = 0; p < num_procs; p++){
      if (recv_flags[p] == 1){
         if (ilower > recvbuf[4*p]){
            ilower = recvbuf[4*p];
         }
         if (iupper < recvbuf[4*p+1]){
            iupper = recvbuf[4*p+1];
         }
         if (jlower > recvbuf[4*p+2]){
            jlower = recvbuf[4*p+2];
         }
         if (jupper < recvbuf[4*p+3]){
            jupper = recvbuf[4*p+3];
         }
      }
   }
   
   free(sendbuf);
   free(recvbuf);
   

   HYPRE_IJMatrixCreate(my_comm, ilower, iupper, jlower, jupper, &ij_matrix);
   HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(ij_matrix);


  // printf("%d, %d\n", ilower, iupper);

   HYPRE_Int num_rows = iupper-ilower+1;
   vector<vector<int>> col_vec(num_rows, vector<int>(0));
   vector<vector<double>> val_vec(num_rows, vector<double>(0));
   for (HYPRE_Int k = 0; k < recvcount; k++){
      HYPRE_Int row_ind = recvbuf_i[k]-ilower;
      col_vec[row_ind].push_back(recvbuf_j[k]);
      val_vec[row_ind].push_back(recvbuf_v[k]);
   }
   for (HYPRE_Int i = 0; i < num_rows; i++){
      int ncols = col_vec[i].size(); 
      int *cols = (int *)calloc(ncols, sizeof(int));
      double *values = (double *)calloc(ncols, sizeof(double));
      for (HYPRE_Int j = 0; j < ncols; j++){
         cols[j] = col_vec[i].back();
         values[j] = val_vec[i].back();
         col_vec[i].pop_back();
         val_vec[i].pop_back();
      }
      int I = ilower+i;
      HYPRE_IJMatrixSetValues(ij_matrix, 1, &ncols, &I, cols, values);
      free(cols);
      free(values);
   }

   HYPRE_IJMatrixAssemble(ij_matrix);
   HYPRE_IJMatrixGetObject(ij_matrix, (void**)B);

  // char buffer[100];
  // sprintf(buffer, "A_async_%d.txt", my_grid);
  // DMEM_PrintParCSRMatrix(*B, buffer);

  // if (my_id == 1){
  //    for (HYPRE_Int k = 0; k < recvcount; k++){
  //       printf("%d, %d: %d %d %e\n", my_id, k, recvbuf_i[k], recvbuf_j[k], recvbuf_v[k]);
  //    }
  //    printf("%d, %d\n", ilower, iupper);

  //   // for (HYPRE_Int k = 0; k < sendcount; k++){
  //   //    printf("%d %d %e\n", sendbuf_i[k], sendbuf_j[k], sendbuf_v[k]);
  //   // }
  //    
  //   // for (HYPRE_Int p = 0; p < num_procs; p++){
  //   //     printf("%d %d\n", sdispls[p], rdispls[p]);
  //   // }
  // }

   free(sendbuf_i);
   free(sendbuf_j);
   free(sendbuf_v);
  // free(recvbuf_j);
  // free(recvbuf_i);
  // free(recvbuf_v);
}

void DMEM_MatrixFromFile(char *mat_file_str, hypre_ParCSRMatrix **A_ptr, MPI_Comm comm)
{
   int num_procs, my_id;
   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);

   idx_t ncon = 1, objval, n;
   int flag = METIS_OK;
   int imbalance = 0;
   int point;
   MetisGraph G;
   Triplet T;
   CSR A, B;
   OrderingData P;

   P.nparts = num_procs;

   if (my_id == 0){
      idx_t nparts = (idx_t)num_procs;
      idx_t options[METIS_NOPTIONS];
      FILE *mat_file = fopen(mat_file_str, "rb");
      ReadBinary_fread_metis(mat_file, &G, &T, 1);
      fclose(mat_file);
      A.n = B.n_glob = G.n;
      A.nnz = G.nnz;
      METIS_SetDefaultOptions(options);
      options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
      idx_t *perm = (idx_t *)calloc(A.n, sizeof(idx_t));
      if (num_procs > 1){
         flag =  METIS_PartGraphKway(&(G.n), &ncon, G.xadj, G.adjncy, NULL, 
                                     NULL, NULL, &nparts, NULL, NULL, 
                                     options, &objval, perm);
         if (flag != METIS_OK){
            printf("****WARNING****: METIS returned error with code %d.\n", flag);
         }
      }
      else {
         for (int i = 0; i < A.n; i++) perm[i] = 0;
      }
      FreeMetis(&G);

      P.perm = (int *)calloc(A.n, sizeof(int));
      P.part = (int *)calloc(P.nparts, sizeof(int));
      for (int i = 0; i < A.n; i++){
         P.perm[i] = perm[i];
         P.part[P.perm[i]]++;
      }
      free(perm);
      P.disp = (int *)malloc((P.nparts+1) * sizeof(int));
      P.disp[0] = 0;
      for (int i = 0; i < P.nparts; i++){
         P.disp[i+1] = P.disp[i] + P.part[i];
      }
      Reorder(&P, &T, &A);
      //WriteCSR(A, "metis_matrix_matlab.txt", 1);
   }
   DMEM_DistributeCSR_RootToFine(A, &B, &P, comm);
   if (my_id == 0){
      FreeTriplet(&T);
      FreeCSR(&A);
   }
  // DMEM_WriteCSR(B, "metis_matrix_matlab.txt", 1, P, comm);
   CSRtoParHypreCSRMatrix(B, A_ptr, P, comm);
   FreeOrdering(&P);
   FreeCSR(&B);
}

void DMEM_DistributeCSR_RootToFine(CSR A,
                                   CSR *B,
                                   OrderingData *P,
                                   MPI_Comm comm)
{
   int num_procs, my_id;
   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);

   MPI_Bcast(&(B->n_glob), 1, MPI_INT, 0, comm);

   int j_ptr_disp;
   P->dispv = (int *)malloc(P->nparts * sizeof(int));
   int *j_ptr_extra = (int *)malloc(P->nparts * sizeof(int));
   if (my_id == 0){
      for (int i = 0; i < P->nparts; i++){ 
         P->dispv[i] = P->disp[i];
         j_ptr_extra[i] = A.j_ptr[P->disp[i+1]];
      }
   }
   else {
      P->part = (int *)malloc(P->nparts * sizeof(int));
      P->disp = (int *)malloc((P->nparts+1) * sizeof(int));
   }
   
   MPI_Bcast(P->part, P->nparts, MPI_INT, 0, comm);
   MPI_Bcast(P->disp, P->nparts+1, MPI_INT, 0, comm);
   MPI_Bcast(P->dispv, P->nparts, MPI_INT, 0, comm);

   B->n = P->part[my_id];

   B->j_ptr = (int *)calloc(B->n+1, sizeof(int));
   MPI_Scatterv(A.j_ptr, 
                P->part, 
                P->dispv, 
                MPI_INT, 
                B->j_ptr, 
                B->n, 
                MPI_INT, 
                0, 
                comm);
   MPI_Scatter(j_ptr_extra, 
               1, 
               MPI_INT, 
               &(B->j_ptr[B->n]), 
               1, 
               MPI_INT,
               0, 
               comm);

   j_ptr_disp = B->j_ptr[0];
   for (int i = 0; i < B->n+1; i++){
      B->j_ptr[i] -= j_ptr_disp; 
   }
  
   int *part_nnz = (int *)malloc(P->nparts * sizeof(int)); 
   int *dispv_nnz = (int *)malloc(P->nparts * sizeof(int)); 
   int count_nnz;
   if (my_id == 0){
      dispv_nnz[0] = 0;
      for (int i = 0; i < P->nparts; i++){
         count_nnz = 0;
         for (int j = P->disp[i]; j < P->disp[i+1]; j++){
            count_nnz += (A.j_ptr[j+1] - A.j_ptr[j]);
         }
         if (i < P->nparts-1) dispv_nnz[i+1] = dispv_nnz[i] + count_nnz;
         part_nnz[i] = count_nnz;
      }
   }

   MPI_Bcast(part_nnz, P->nparts, MPI_INT, 0, comm);
   MPI_Bcast(dispv_nnz, P->nparts, MPI_INT, 0, comm);

   B->nnz = part_nnz[my_id];
   B->i = (int *)calloc(B->nnz, sizeof(int));
   B->val = (double *)calloc(B->nnz, sizeof(double));
 
   MPI_Scatterv(A.i, part_nnz, dispv_nnz, MPI_INT,
                B->i, B->nnz, MPI_INT, 0, comm);
   MPI_Scatterv(A.val, part_nnz, dispv_nnz, MPI_DOUBLE,
                B->val, B->nnz, MPI_DOUBLE, 0, comm);

   free(part_nnz);
   free(dispv_nnz);
   free(j_ptr_extra);
}

void ReadBinary_fread_metis(FILE *mat_file,
                            MetisGraph *G,
                            Triplet *T,
                            int symm_flag)
{
   using namespace std;
   size_t size;
   int temp_size;
   int k, q;
   int row, col;
   double elem;
   Triplet_AOS *buffer;

   fseek(mat_file, 0, SEEK_END);
   size = ftell(mat_file);
   rewind(mat_file);
   buffer = (Triplet_AOS *)malloc(sizeof(Triplet_AOS) * size);
   fread(buffer, sizeof(Triplet_AOS), size, mat_file);

   int file_lines = size/sizeof(Triplet_AOS);
   int max_row = 0;
   T->nnz = 0;
   for (int k = 0; k < file_lines; k++){
      row = buffer[k].i;
      col = buffer[k].j;
      elem = buffer[k].val;

      if (fabs(elem) > 0){
         if (row > max_row){
            max_row = row;
         }
         T->nnz += 1;
         if (symm_flag == 1 && row != col){
            T->nnz += 1;
         }
      }
   }
   G->nnz = T->nnz;
   G->n = T->n = max_row;

   vector<list<int>> col_list(T->n);
   vector<list<double>> elem_list(T->n);
   for (int k = 0; k < file_lines; k++){
      row = buffer[k].i;
      col = buffer[k].j;
      elem = buffer[k].val;

      if (fabs(elem) > 0){
         col_list[row-1].push_back(col-1);
         elem_list[row-1].push_back(elem);
         if (symm_flag == 1 && row != col){
            col_list[col-1].push_back(row-1);
            elem_list[col-1].push_back(elem);
         }
      }
   }

   G->xadj = (idx_t *)malloc(((int)G->n+1) * sizeof(idx_t));
   G->adjncy = (idx_t *)malloc((int)G->nnz * sizeof(idx_t));
   G->adjwgt = (real_t *)malloc((int)G->nnz * sizeof(real_t));

   T->i = (int *)malloc(T->nnz * sizeof(int));
   T->j = (int *)malloc(T->nnz * sizeof(int));
   T->val = (double *)malloc(T->nnz * sizeof(double));

   List_to_Metis(G, T, col_list, elem_list);
   free(buffer);
}

void ReorderTriplet(Triplet T, CSR *A, OrderingData *P)
{
   int row, col;
   int s, q, k, map_i, p;
   double elem;

   int n = A->n;
   int nnz = A->nnz;
   int **col_list = (int **)malloc(n * sizeof(int *));
   double **elem_list = (double **)malloc(n * sizeof(double *));
   int *len = (int *)calloc(n, sizeof(int));
   int *col_flag = (int *)calloc(n, sizeof(int));
   int *p_count = (int *)calloc(P->nparts, sizeof(int));
   A->j_ptr = (int *)calloc(n+1, sizeof(int));
   P->map = (int *)calloc(n, sizeof(int));
   for (int i = 0; i < nnz; i++){
      col = T.j[i];
      row = T.i[i];
      elem = T.val[i];
      p = P->perm[col];
      k = p_count[p];
      if (!col_flag[col]){
         P->map[col] = k + P->disp[p];
         col_flag[col] = 1;
         p_count[p]++;
      }
      map_i = P->map[col];
      len[map_i]++;
   }
   for (int i = 0; i < n; i++){
      col_list[i] = (int *)calloc(len[i], sizeof(int));
      elem_list[i] = (double *)calloc(len[i], sizeof(double));
      len[i] = 0;
   }
   for (int i = 0; i < nnz; i++){
      col = T.j[i];
      row = T.i[i];
      elem = T.val[i];
      p = P->perm[col];
      map_i = P->map[col];
      q = len[map_i];
      col_list[map_i][q] = row;
      elem_list[map_i][q] = elem;
      len[map_i]++;
   }
   A->i = (int *)calloc(nnz,  sizeof(int));
   A->val = (double *)calloc(nnz,  sizeof(double));
   k = 0;
   A->j_ptr[0] = 0;
   for (int i = 0; i < n; i++){
       A->j_ptr[i+1] = A->j_ptr[i] + len[i];
       for (int j = 0; j < len[i]; j++){
          A->i[k] = col_list[i][j];
          A->val[k] = elem_list[i][j];
          k++;
       }
   }
   for (int i = 0; i < n; i++){
      for (int j = A->j_ptr[i]; j < A->j_ptr[i+1]; j++){
         A->i[j] = P->map[A->i[j]];
      }
   }
   for (int i = 0; i < n; i++){
      free(col_list[i]);
      free(elem_list[i]);
   }
   free(col_list);
   free(elem_list);
   free(p_count);
   free(col_flag);
   free(len);
}

void Reorder(OrderingData *P, 
             Triplet *T,
             CSR *A)
{
   ReorderTriplet(*T, A, P);
}

void List_to_Metis(MetisGraph *G,
                   Triplet *T,
                   std::vector<std::list<int>> col_list,
                   std::vector<std::list<double>> elem_list)
{
   int temp_size;
   int k = 0;
   G->xadj[0] = 0;
   for (int i = 0; i < G->n; i++){
      col_list[i].begin();
      elem_list[i].begin();
      temp_size = col_list[i].size();
      G->xadj[i+1] = G->xadj[i] + (idx_t)temp_size;
      for (int j = 0; j < temp_size; j++){
         G->adjncy[k] = col_list[i].front();
         col_list[i].pop_front();
         G->adjwgt[k] = elem_list[i].front();
         elem_list[i].pop_front();

         T->j[k] = i;
         T->i[k] = G->adjncy[k];
         T->val[k] = G->adjwgt[k];

         k++;
      }
   }
}

void CSRtoParHypreCSRMatrix(CSR B,
                            hypre_ParCSRMatrix **A_ptr,
                            OrderingData P,
                            MPI_Comm comm)
{
   int num_procs, my_id;
   MPI_Comm_rank(comm, &my_id);
   MPI_Comm_size(comm, &num_procs);

   HYPRE_IJMatrix Aij;
   HYPRE_IJMatrixCreate(comm, P.disp[my_id], P.disp[my_id+1]-1, P.disp[my_id], P.disp[my_id+1]-1, &Aij);
   HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(Aij);

   for (int i = 0; i < B.n; i++){
      int nnz = B.j_ptr[i+1] - B.j_ptr[i];
      double *values = (double *)malloc(nnz * sizeof(double));
      int *cols = (int *)malloc(nnz * sizeof(int));

      int jj = 0;
      for (int j = B.j_ptr[i]; j < B.j_ptr[i+1]; j++){
         cols[jj] = B.i[j];
         values[jj] = B.val[j];
         jj++;
      }

      int row_glob = P.disp[my_id]+i;
      /* Set the values for row i */
      HYPRE_IJMatrixSetValues(Aij, 1, &nnz, &row_glob, cols, values);
   }

   HYPRE_IJMatrixAssemble(Aij);
   void *object;
   HYPRE_IJMatrixGetObject(Aij, &object);
   *A_ptr = (hypre_ParCSRMatrix *)object;
}
