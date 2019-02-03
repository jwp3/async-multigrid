#!/bin/bash

smoother=${1}
num_threads=${2}
problem=${3}
num_smooth_sweeps=${4}
n=${5}
coarsen_type=10
agg_nl=2
interp_type=6

./ResHist.sh \
async_multadd \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"global" \
"global" \

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
"global" \

./ResHist.sh \
async_multadd \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"global" \
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
"local" \

./ResHist.sh \
async_afacx \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${n} \
${problem} \
${interp_type} \
"global" \
"local" \

./ResHist.sh \
async_afacx \
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
"local" \

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
"local" \

./ResHist.sh \
afacx \
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
"local" \

./ResHist.sh \
afacx \
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
"local" \

./ResHist.sh \
multadd \
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
"local" \
