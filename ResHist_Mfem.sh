#!/bin/bash

solver=${1}
smoother=${2}
num_threads=${3}
async_type=${4}
coarsen_type=${5}
agg_nl=1

file_directory="data/ResHist_Mfem/"
file_name="${file_directory}SOLVER-${solver}-SMOOTHER-${smoother}-PROBLEM-${problem}-NUMTHREADS-${num_threads}-ASYNCTYPE-${async_type}-COARSENTYPE-${coarsen_type}-AGGNL-${agg_nl}-REFLEVELS-${ref_levels}"
mkdir -p ${file_directory}
rm -f ${file_name}

ref_levels=5
problem="mfem"
num_runs=20
num_cycles=50
start_cycle=1

	./Main -format_output -num_runs ${num_runs} -num_cycles ${num_cycles} -start_cycle ${start_cycle} -mfem_ref_levels ${ref_levels} -num_threads ${num_threads} -solver ${solver} -smoother ${smoother} -problem ${problem} -async_type ${async_type} -agg_nl ${agg_nl} | tee -a ${file_name}
