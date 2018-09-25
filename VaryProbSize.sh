#!/bin/bash

solver=${1}
smoother=${2}
problem=${3}
num_threads=${4}
async_type=${5}
coarsen_type=${6}
agg_nl=0
num_smooth_sweeps=1

file_directory="data/VaryProbSize/"
file_name="${file_directory}SOLVER-${solver}-SMOOTHER-${smoother}-SWEEPS-${num_smooth_sweeps}-PROBLEM-${problem}-NUMTHREADS-${num_threads}-ASYNCTYPE-${async_type}-COARSENTYPE-${coarsen_type}-AGGNL-${agg_nl}-"
mkdir -p ${file_directory}
rm -f ${file_name}

num_runs=20
num_cycles=50
#n_list=(10 20 50 100 150)
n_list=(65 129 257 513 1025)

for n in ${n_list[@]}
do
	KMP_AFFINITY=compact ./Main -format_output -no_reshist -smoother ${smoother} -num_smooth_sweeps ${num_smooth_sweeps} -num_runs ${num_runs} -n ${n} -num_cycles ${num_cycles} -num_threads ${num_threads} -solver ${solver} -problem ${problem} -async_type ${async_type} -converge_test_type all | tee -a ${file_name}
done
