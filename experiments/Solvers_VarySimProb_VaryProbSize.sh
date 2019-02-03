#!/bin/bash

smoother="j"
problem="7pt"
coarsen_type=10
agg_nl=0
interp_type=6

./VarySimProb_VaryProbSize.sh \
async_multadd \
semi \
${problem} \
${smoother} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"sol" \

./VarySimProb_VaryProbSize.sh \
async_afacx \
semi \
${problem} \
${smoother} \
${coarsen_type} \
${agg_nl} \
${interp_type} \
"local" \
"sol" \
