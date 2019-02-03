#!/bin/bash

solver=${1}
async_type=${2}
problem=${3}
smoother=${4}
coarsen_type=${5}
agg_nl=${6}
interp_type=${7}
converge_test_type=${8}
read_type=${9}

file_directory="data/VarySimDelay_VaryProbSize/"
file_name="${file_directory}"\
"SOLVER-${solver}-"\
"SMOOTHER-${smoother}-"\
"PROBLEM-${problem}-"\
"ASYNCTYPE-${async_type}-"\
"COARSENTYPE-${coarsen_type}-"\
"INTERPTYPE-${interp_type}-"\
"AGGNL-${agg_nl}-"\
"CONVERGETYPE-${converge_test_type}-"\
"READTYPE-${read_type}-"\

mkdir -p ${file_directory}
rm -f ${file_name}

smooth_weight=.9
num_runs=50
num_cycles=20
n_list=(20 30 40 50 60)
d_list=(1 2 3 5 20)
p=".1"

for d in ${d_list[@]}
do
	for n in ${n_list[@]}
	do
		./Main \
		-format_output \
		-smoother ${smoother} \
		-num_runs ${num_runs} \
		-n ${n} \
		-num_cycles ${num_cycles} \
		-smooth_weight ${smooth_weight} \
		-num_threads 1 \
		-solver ${solver} \
		-problem ${problem} \
		-async_type ${async_type} \
		-converge_test_type ${converge_test_type} \
		-interp_type ${interp_type} \
		-agg_nl ${agg_nl} \
		-sim_update_prob ${p} \
		-sim_read_delay ${d} \
		-read_type ${read_type} \
		| tee -a ${file_name}
	done
done

for n in ${n_list[@]}
do
	./Main \
	-format_output \
	-smoother ${smoother} \
	-num_runs 1 \
	-n ${n} \
	-num_cycles ${num_cycles} \
	-smooth_weight ${smooth_weight} \
	-num_threads 1 \
	-solver ${solver} \
	-problem ${problem} \
	-async_type ${async_type} \
	-converge_test_type ${converge_test_type} \
	-interp_type ${interp_type} \
	-agg_nl ${agg_nl} \
	-sim_update_prob 1 \
	-sim_read_delay 0 \
	-read_type ${read_type} \
	| tee -a ${file_name}
done
