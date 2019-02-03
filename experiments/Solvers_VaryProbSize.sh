#!/bin/bash

smoother=${1}
num_threads=${2}
problem=${3}
num_smooth_sweeps=${4}
coarsen_type=10
agg_nl=1
interp_type=0

./VaryProbSize.sh \
async_multadd \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"global" \

./VaryProbSize.sh \
async_multadd \
semi \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"global" \

./VaryProbSize.sh \
async_multadd \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \

./VaryProbSize.sh \
async_multadd \
semi \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \

./VaryProbSize.sh \
async_afacx \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \

./VaryProbSize.sh \
async_afacx \
semi \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \

./VaryProbSize.sh \
multadd \
semi \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \

./VaryProbSize.sh \
multadd \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \

./VaryProbSize.sh \
afacx \
semi \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \

./VaryProbSize.sh \
afacx \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \

./VaryProbSize.sh \
mult \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"local" \
