#!/bin/bash

smoother="j"
problem="7pt"
coarsen_type=10
agg_nl=0
interp_type=6

./VarySimDelay_VaryProbSize.sh \
async_multadd \
full \
${problem} \
${smoother} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"sol" \

./VarySimDelay_VaryProbSize.sh \
async_multadd \
full \
${problem} \
${smoother} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"res" \

./VarySimDelay_VaryProbSize.sh \
async_afacx \
full \
${problem} \
${smoother} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"sol" \

./VarySimDelay_VaryProbSize.sh \
async_afacx \
full \
${problem} \
${smoother} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"res" \
