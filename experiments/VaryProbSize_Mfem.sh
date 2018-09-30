#!/bin/bash

solver=${1}
async_type=${2}
problem=${3}
smoother=${4}
num_threads=${5}
num_smooth_sweeps=${6}
coarsen_type=${7}
agg_nl=${8}
mfem_mesh_file=${9}
interp_type=${10}
converge_test_type=${11}

file_directory="data/VaryProbSize_Mfem/"
file_name="${file_directory}"\
"SOLVER-${solver}-"\
"SMOOTHER-${smoother}-"\
"SWEEPS-${num_smooth_sweeps}-"\
"PROBLEM-${problem}-"\
"NUMTHREADS-${num_threads}-"\
"ASYNCTYPE-${async_type}-"\
"COARSENTYPE-${coarsen_type}-"\
"INTERPTYPE-${interp_type}-"\
"AGGNL-${agg_nl}-"\
"CONVERGETYPE-${converge_test_type}-"\

mkdir -p ${file_directory}
rm -f ${file_name}

smooth_weight=1
num_runs=20
num_cycles=20
ref_list=(1 2 3 4 5)

for ref in ${ref_list[@]}
do
	KMP_AFFINITY=compact ./Main \
		-format_output \
		-smoother ${smoother} \
		-num_smooth_sweeps ${num_smooth_sweeps} \
		-num_runs ${num_runs} \
		-mfem_ref_levels ${ref} \
		-num_cycles ${num_cycles} \
		-num_threads ${num_threads} \
		-solver ${solver} \
		-problem ${problem} \
		-mfem_mesh_file ${mfem_mesh_file} \
		-smooth_weight ${smooth_weight} \
		-async_type ${async_type} \
		-converge_test_type ${converge_test_type} \
		-interp_type ${interp_type} \
		| tee -a ${file_name}
done
