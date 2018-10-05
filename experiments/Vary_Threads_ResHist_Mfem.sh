#!/bin/bash

num_smooth_sweeps=1
ref_levels=3
t_list=(1 2 4 8 16 32)



problem="mfem_laplace"
mfem_mesh_file="../mfem/data/ball-nurbs.mesh"

smoother="L1j"
for t in ${t_list[@]}
do
	echo "THREADS ${t}"
	num_threads=${t}
	./Threads_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
done

smoother="async_gs"
for t in ${t_list[@]}
do
        echo "THREADS ${t}"
        num_threads=${t}
        ./Threads_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
done



problem="mfem_elast"
mfem_mesh_file="../mfem/data/beam-tet.mesh"

smoother="L1j"
for t in ${t_list[@]}
do
        echo "THREADS ${t}"
        num_threads=${t}
        ./Threads_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
done

smoother="async_gs"
for t in ${t_list[@]}
do
        echo "THREADS ${t}"
        num_threads=${t}
        ./Threads_ResHist_Mfem.sh ${smoother} ${num_threads} ${problem} ${mfem_mesh_file} ${num_smooth_sweeps} ${ref_levels}
done
