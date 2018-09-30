#!/bin/bash

smoother=${1}
num_threads=${2}
problem=${3}
num_smooth_sweeps=${4}
n=${5}
coarsen_type=10
agg_nl=1
interp_type=6

./ResHist.sh \
async_multadd \
semi \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"local" \

./ResHist.sh \
async_multadd \
semi \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"global" \

./ResHist.sh \
mult \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"local" \

./ResHist.sh \
multadd \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"local" \
