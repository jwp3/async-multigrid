#!/bin/bash

num_threads=32
num_smooth_sweeps=1
ref_levels=4

problem="mfem_elast"
mfem_mesh_file="./mfem/data/beam-tet.mesh"
smoother="L1j"
./Solvers_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels} 
smoother="j"
./Solvers_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
smoother="hybrid_jgs"
./Solvers_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
smoother="async_gs"
./Solvers_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}

problem="mfem_laplace"
mfem_mesh_file="./mfem/data/ball-nurbs.mesh"
smoother="L1j"
./Solvers_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
smoother="j"
./Solvers_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
smoother="hybrid_jgs"
./Solvers_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
smoother="async_gs"
./Solvers_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
