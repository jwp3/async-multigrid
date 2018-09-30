#!/bin/bash

problem="mfem_laplace"
mfem_mesh_file="./mfem/data/ball-nurbs.mesh"
num_smooth_sweeps=1
smoother="L1j"
ref_levels=3

t_list=(1 2 4 8 16 32)

for t in ${t_list[@]}
do
	echo "THREADS ${t}"
	num_threads=${t}
	./Threads_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
done
