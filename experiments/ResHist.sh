#!/bin/bash

solver=${1}
async_type=${2}
smoother=${3}
num_threads=${4}
num_smooth_sweeps=${5}
coarsen_type=${6}
agg_nl=${7}
n=${8}
problem=${9}
interp_type=${10}
converge_test_type=${11}
res_compute_type=${12}

if [ "${smoother}" = "L1j" ]; then
        if [ "${solver}" = "async_multadd" ]; then
                if [ "${res_compute_type}" = "global" ]; then
                        exit
                fi
        fi
        if [ "${solver}" = "afacx" ] || [ "${solver}" = "async_afacx" ]; then
                exit
        fi
fi

file_directory="data/ResHist/"
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
"N-${n}-"\
"CONVERGETYPE-${converge_test_type}-"\
"RESTYPE-${res_compute_type}-"\

mkdir -p ${file_directory}
rm -f ${file_name}

smooth_weight=.9
num_runs=20
num_cycles=200
start_cycle=5
incr_cycle=5

#runs_list=($(seq 1 1 ${num_runs}))
#cycles_list=($(seq ${start_cycle} ${incr_cycle} ${num_cycles}))
#
#for cycles in ${cycles_list[@]}
#do
#	for run in ${runs_list[@]}
#	do 
#		./Main -format_output \
#		-num_cycles ${cycles} \
#		-n ${n} \
#		-num_threads ${num_threads} \
#		-solver ${solver} \
#		-smoother ${smoother} \
#		-problem ${problem} \
#		-async_type ${async_type} \
#		-agg_nl ${agg_nl} \
#		-smooth_weight ${smooth_weight} \
#		-converge_test_type ${converge_test_type} \
#		-num_smooth_sweeps ${num_smooth_sweeps} \
#		-coarsen_type ${coarsen_type} \
#		-mfem_mesh_file ${mfem_mesh_file} \
#		-interp_type ${interp_type} \
#		-res_compute_type ${res_compute_type} \
#		| tee -a ${file_name}
#	done
#done

KMP_AFFINITY=compact ./Main -format_output \
	-num_cycles ${num_cycles} \
	-n ${n} \
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
	-res_compute_type ${res_compute_type} \
	-num_runs ${num_runs} \
	-num_cycles ${num_cycles} \
	-start_cycle ${start_cycle} \
	-incr_cycle ${incr_cycle} \
	| tee -a ${file_name} \