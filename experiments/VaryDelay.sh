#!/bin/bash

solver=${1}
async_type=${2}
smoother=${3}
num_smooth_sweeps=${4}
coarsen_type=${5}
agg_nl=${6}
n=${7}
problem=${8}
interp_type=${9}
converge_test_type=${10}

file_directory="data/VaryDelay/"
file_name="${file_directory}"\
"SOLVER-${solver}-"\
"SMOOTHER-${smoother}-"\
"SWEEPS-${num_smooth_sweeps}-"\
"PROBLEM-${problem}-"\
"ASYNCTYPE-${async_type}-"\
"COARSENTYPE-${coarsen_type}-"\
"INTERPTYPE-${interp_type}-"\
"AGGNL-${agg_nl}-"\
"N-${n}-"\
"CONVERGETYPE-${converge_test_type}-"\

mkdir -p ${file_directory}
rm -f ${file_name}

num_threads=1
smooth_weight=1
num_runs=20
num_cycles=20
grid_wait=100
delay_list=(0 $(seq 10 10 ${grid_wait}))

for delay in ${delay_list[@]}
do
	./Main \
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
	-sim_read_delay ${delay} \
	-sim_grid_wait ${grid_wait} \
	| tee -a ${file_name}
done
