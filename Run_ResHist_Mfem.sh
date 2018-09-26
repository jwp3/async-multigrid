#!/bin/bash

smoother="async_gs"
num_threads=272
num_smooth_sweeps=4
coarsen_type="10"
agg_nl=2
ref_levels=5
problem="mfem_elast"
mfem_mesh_file="./mfem/data/beam-tet.mesh"

./ResHist_Mfem.sh \
async_multadd \
${smoother} \
${num_threads} \
full \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \

./ResHist_Mfem.sh \
async_multadd \
${smoother} \
${num_threads} \
semi \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \

./ResHist_Mfem.sh \
async_afacx \
${smoother} \
${num_threads} \
full \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \

./ResHist_Mfem.sh \
async_afacx \
${smoother} \
${num_threads} \
semi \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \

./ResHist_Mfem.sh \
afacx \
${smoother} \
${num_threads} \
full \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \

./ResHist_Mfem.sh \
mult \
${smoother} \
${num_threads} \
full \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \

./ResHist_Mfem.sh \
multadd \
${smoother} \
${num_threads} \
full \
${num_smooth_sweeps} \
${coarsen_type} \
${agg_nl} \
${ref_levels} \
${problem} \
${mfem_mesh_file} \
