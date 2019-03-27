#include "Main.hpp"
#include "DMEM_Main.hpp"

using namespace std;
using namespace mfem;

void DMEM_ParMfem(DMEM_AllData *dmem_all_data,
                  HYPRE_ParCSRMatrix *A,
                  MPI_Comm comm)

{
   int num_procs, my_id;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);

   Mesh *mesh = new Mesh(dmem_all_data->mfem.mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int l = 0; l < dmem_all_data->mfem.ref_levels; l++){
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(dmem_all_data->mfem.order, 1));
   }

   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;
   {
      for (int l = 0; l < dmem_all_data->mfem.par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   FiniteElementCollection *fec;
   fec = new H1_FECollection(dmem_all_data->mfem.order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   ParGridFunction x(fespace);
   x = 0.0;

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();

   HypreParMatrix A_mfem;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A_mfem, X, B);

   *A = A_mfem.GetHypreMatrix();
  // for (int p = 0; p < num_procs; p++){
  //    if (my_id == p){
  //    }
  //    MPI_Barrier(comm);
  // }

  // delete a;
  // delete b;
  // delete fespace;
  // delete fec;
  // delete pmesh;
   return;
}
