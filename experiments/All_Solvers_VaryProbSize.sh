#!/bin/bash

num_threads=32
num_smooth_sweeps=1

smoother="async_gs"
problem="27pt"
./Solvers_VaryProbSize.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps}
problem="mfem_elast"
mfem_mesh_file="./mfem/data/beam-tet.mesh"
./Solvers_VaryProbSize_Mfem.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${mfem_mesh_file}
#problem="mfem_laplace"
#mfem_mesh_file="./mfem/data/ball-nurbs.mesh"
#./Solvers_VaryProbSize_Mfem.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${mfem_mesh_file}

smoother="j"
problem="27pt"
./Solvers_VaryProbSize.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps}
problem="mfem_elast"
mfem_mesh_file="./mfem/data/beam-tet.mesh"
./Solvers_VaryProbSize_Mfem.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${mfem_mesh_file}
#smoother="L1j"
#problem="mfem_laplace"
#mfem_mesh_file="./mfem/data/ball-nurbs.mesh"
#./Solvers_VaryProbSize_Mfem.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${mfem_mesh_file}
