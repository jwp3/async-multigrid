#!/bin/bash

solver=${1}
smoother=${2}
problem=${3}
num_threads=${4}
async_type=${5}
coarsen_type=${6}
agg_nl=${7}

n=1000
ref_levels=4
num_cycles=10

file_directory="data/VaryThreads/"
file_name="${file_directory}SOLVER-${solver}-SMOOTHER-${smoother}-PROBLEM-${problem}-CYCLES-${num_cycles}-ASYNCTYPE-${async_type}-COARSENTYPE-${coarsen_type}-AGGNL-${agg_nl}-"
mkdir -p ${file_directory}
rm -f ${file_name}

num_runs=20
t_list=(1 2 4 8 16 32)

for t in ${t_list[@]}
do
	num_threads=${t}
	./Main -format_output -no_reshist -num_runs ${num_runs} -n ${n} -mfem_ref_levels ${ref_levels} -num_cycles ${num_cycles} -num_threads ${num_threads} -solver ${solver} -problem ${problem} -async_type ${async_type} -converge_test_type all | tee -a ${file_name}
done
