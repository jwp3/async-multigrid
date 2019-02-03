#!/bin/bash

num_smooth_sweeps=1

n=30
problem="7pt"
smoother="j"
./Solvers_VaryDelay.sh ${smoother} ${problem} ${num_smooth_sweeps} ${n}

ref_levels=3
problem="mfem_elast"
mfem_mesh_file="../mfem/data/beam-tet.mesh"
smoother="j"
./Solvers_VaryDelay_Mfem.sh ${smoother} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels} 
