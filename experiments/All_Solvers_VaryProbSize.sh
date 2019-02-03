#!/bin/bash

num_threads=32
num_smooth_sweeps=1

problem="27pt"
smoother="async_gs"
./Solvers_VaryProbSize.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps}
smoother="j"
./Solvers_VaryProbSize.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps}

problem="7pt"
smoother="async_gs"
./Solvers_VaryProbSize.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps}
smoother="j"
./Solvers_VaryProbSize.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps}

#problem="mfem_elast"
#mfem_mesh_file="../mfem/data/beam-tet.mesh"
#smoother="async_gs"
#./Solvers_VaryProbSize_Mfem.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${mfem_mesh_file}
#smoother="j"
#./Solvers_VaryProbSize_Mfem.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${mfem_mesh_file}
