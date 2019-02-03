#!/bin/bash

smoother=${1}
num_threads=${2}
problem=${3}
mfem_mesh_file=${4}
num_smooth_sweeps=${5}
ref_levels=${6}
coarsen_type=10
agg_nl=2
interp_type=6

./ResHist_Mfem.sh \
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
"global" \

./ResHist_Mfem.sh \
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
"global" \

./ResHist_Mfem.sh \
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
"local" \

./ResHist_Mfem.sh \
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
"local" \

./ResHist_Mfem.sh \
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
"local" \

./ResHist_Mfem.sh \
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
"local" \

./ResHist_Mfem.sh \
mult \
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
"local" \

./ResHist_Mfem.sh \
afacx \
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
"local" \

./ResHist_Mfem.sh \
afacx \
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
"local" \

./ResHist_Mfem.sh \
multadd \
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
"local" \

./ResHist_Mfem.sh \
multadd \
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
"local" \
