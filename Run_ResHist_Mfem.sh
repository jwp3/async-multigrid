#!/bin/bash

num_threads=272

./ResHist_Mfem.sh \
async_multadd \
symm_j \
${num_threads} \
full \

./ResHist_Mfem.sh \
async_multadd \
symm_j \
${num_threads} \
semi \

./ResHist_Mfem.sh \
async_multadd \
semi_async_gs \
${num_threads} \
full \

./ResHist_Mfem.sh \
async_multadd \
semi_async_gs \
${num_threads} \
semi \

./ResHist_Mfem.sh \
async_afacx \
semi_async_gs \
${num_threads} \
full \

./ResHist_Mfem.sh \
async_afacx \
semi_async_gs \
${num_threads} \
semi \

./ResHist_Mfem.sh \
afacx \
semi_async_gs \
${num_threads} \
full \

./ResHist_Mfem.sh \
mult \
semi_async_gs \
${num_threads} \
full \

./ResHist_Mfem.sh \
afacx \
hybrid_jgs \
${num_threads} \
full \
