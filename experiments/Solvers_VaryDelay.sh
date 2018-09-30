#!/bin/bash

smoother=${1}
problem=${2}
num_smooth_sweeps=${3}
n=${4}
coarsen_type="10"
agg_nl=1
interp_type=6

./VaryDelay.sh \
async_multadd \
full \
${smoother} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"local" \

./VaryDelay.sh \
async_multadd \
full \
${smoother} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"global" \

./VaryDelay.sh \
async_multadd \
semi \
${smoother} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"local" \

./VaryDelay.sh \
async_multadd \
semi \
${smoother} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"global" \

./VaryDelay.sh \
async_afacx \
full \
${smoother} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"local" \

./VaryDelay.sh \
async_afacx \
full \
${smoother} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"global" \

./VaryDelay.sh \
async_afacx \
semi \
${smoother} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"local" \

./VaryDelay.sh \
async_afacx \
semi \
${smoother} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"global" \
