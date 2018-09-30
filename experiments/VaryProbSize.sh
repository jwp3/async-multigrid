#!/bin/bash

solver=${1}
async_type=${2}
problem=${3}
smoother=${4}
num_threads=${5}
num_smooth_sweeps=${6}
coarsen_type=${7}
agg_nl=${8}
interp_type=${9}
converge_test_type=${10}

file_directory="data/VaryProbSize/"
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
n_list=(30 40 50 60 70)

for n in ${n_list[@]}
do
	KMP_AFFINITY=compact ./Main \
		-format_output \
		-smoother ${smoother} \
		-num_smooth_sweeps ${num_smooth_sweeps} \
		-num_runs ${num_runs} \
		-n ${n} \
		-num_cycles ${num_cycles} \
		-smooth_weight ${smooth_weight} \
		-num_threads ${num_threads} \
		-solver ${solver} \
		-problem ${problem} \
		-async_type ${async_type} \
		-converge_test_type ${converge_test_type} \
		-interp_type ${interp_type} \
		-agg_nl ${agg_nl} \
		| tee -a ${file_name}
done
