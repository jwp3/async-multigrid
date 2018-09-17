#!/bin/bash

solver=${1}
smoother=${2}
problem=${3}
num_threads=${4}
async_type=${5}
coarsen_type=${6}
agg_nl=0

file_directory="data/VaryProbSize/"
file_name="${file_directory}SOLVER-${solver}-SMOOTHER-${smoother}-PROBLEM-${problem}-NUMTHREADS-${num_threads}-ASYNCTYPE-${async_type}-COARSENTYPE-${coarsen_type}-AGGNL-${agg_nl}-"
mkdir -p ${file_directory}
rm -f ${file_name}

num_runs=20
num_cycles=20
#n_list=(10 20 50 100 150)
n_list=(100 200 500 1000 1500)

for n in ${n_list[@]}
do
	KMP_AFFINITY=balanced,granularity=fine ./Main -format_output -num_runs ${num_runs} -n ${n} -num_cycles ${num_cycles} -num_threads ${num_threads} -solver ${solver} -problem ${problem} -async_type ${async_type} -converge_test_type all | tee -a ${file_name}
done
