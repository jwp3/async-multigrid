#!/bin/bash

solver=${1}
smoother=${2}
num_threads=${3}
async_type=${4}
num_smooth_sweeps=${5}
coarsen_type=${6}
agg_nl=${7}
ref_levels=${8}
problem=${9}
mfem_mesh_file=${10}

file_directory="data/ResHist_Mfem/"
file_name="${file_directory}SOLVER-${solver}-SMOOTHER-${smoother}-SWEEPS-${num_smooth_sweeps}-PROBLEM-${problem}-NUMTHREADS-${num_threads}-ASYNCTYPE-${async_type}-COARSENTYPE-${coarsen_type}-AGGNL-${agg_nl}-REFLEVELS-${ref_levels}-"
mkdir -p ${file_directory}
rm -f ${file_name}

num_runs=20
num_cycles=50
start_cycle=2
incr_cycle=2

KMP_AFFINITY=compact ./Main -format_output \
	-num_runs ${num_runs} \
	-num_cycles ${num_cycles} \
	-start_cycle ${start_cycle} \
	-incr_cycle ${incr_cycle} \
	-mfem_ref_levels ${ref_levels} \
	-num_threads ${num_threads} \
	-solver ${solver} \
	-smoother ${smoother} \
	-problem ${problem} \
	-async_type ${async_type} \
	-agg_nl ${agg_nl} \
	-smooth_weight 1 \
	-converge_test_type all \
	-num_smooth_sweeps ${num_smooth_sweeps} \
	-coarsen_type ${coarsen_type} \
	-mfem_mesh_file ${mfem_mesh_file} \
	| tee -a ${file_name} \
