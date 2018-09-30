#!/bin/bash

smoother=${1}
problem=${2}
mfem_mesh_file=${3}
num_smooth_sweeps=${4}
ref_levels=${5}
coarsen_type="10"
agg_nl=1
interp_type=6

./VaryDelay_Mfem.sh \
async_multadd \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
${interp_type} \
"local" \

./VaryDelay_Mfem.sh \
async_multadd \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
${interp_type} \
"global" \

./VaryDelay_Mfem.sh \
async_multadd \
semi \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
${interp_type} \
"local" \

./VaryDelay_Mfem.sh \
async_multadd \
semi \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
${interp_type} \
"global" \

./VaryDelay_Mfem.sh \
async_afacx \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
${interp_type} \
"local" \

./VaryDelay_Mfem.sh \
async_afacx \
full \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
${interp_type} \
"global" \

./VaryDelay_Mfem.sh \
async_afacx \
semi \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
${interp_type} \
"local" \

./VaryDelay_Mfem.sh \
async_afacx \
semi \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
${interp_type} \
"global" \
