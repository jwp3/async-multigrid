#!/bin/bash

solver=${1}
async_type=${2}
smoother=${3}
num_threads=${4}
num_smooth_sweeps=${5}
coarsen_type=${6}
agg_nl=${7}
ref_levels=${8}
problem=${9}
mfem_mesh_file=${10}
interp_type=${11}
converge_test_type=${12}

file_directory="data/ResHist_Mfem/"
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
"REFLEVELS-${ref_levels}-"\
"CONVERGETYPE-${converge_test_type}-"\

mkdir -p ${file_directory}
rm -f ${file_name}

smooth_weight=.5
num_runs=20
num_cycles=200
start_cycle=4
incr_cycle=4

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
	-smooth_weight ${smooth_weight} \
	-converge_test_type ${converge_test_type} \
	-num_smooth_sweeps ${num_smooth_sweeps} \
	-coarsen_type ${coarsen_type} \
	-mfem_mesh_file ${mfem_mesh_file} \
	-interp_type ${interp_type} \
	| tee -a ${file_name} \
