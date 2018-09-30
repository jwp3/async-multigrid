#!/bin/bash

smoother=${1}
num_threads=${2}
problem=${3}
num_smooth_sweeps=${4}
mfem_mesh_file=${5}
coarsen_type=10
agg_nl=1
interp_type=0

./VaryProbSize_Mfem.sh \
async_multadd \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${mfem_mesh_file} \
${interp_type} \
"global" \

./VaryProbSize_Mfem.sh \
async_multadd \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${mfem_mesh_file} \
${interp_type} \
"local" \

./VaryProbSize_Mfem.sh \
async_multadd \
semi \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${mfem_mesh_file} \
${interp_type} \
"global" \

./VaryProbSize_Mfem.sh \
async_multadd \
semi \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${mfem_mesh_file} \
${interp_type} \
"local" \

./VaryProbSize_Mfem.sh \
mult \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${mfem_mesh_file} \
${interp_type} \
"local" \

./VaryProbSize_Mfem.sh \
multadd \
full \
${problem} \
${smoother} \
${num_threads} \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${mfem_mesh_file} \
${interp_type} \
"local" \
