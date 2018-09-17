#!/bin/bash

num_threads=136
smoother="async_gs"

./ResHist_Mfem.sh \
async_multadd \
${smoother} \
${num_threads} \
full \

./ResHist_Mfem.sh \
async_multadd \
${smoother} \
${num_threads} \
semi \

./ResHist_Mfem.sh \
async_afacx \
${smoother} \
${num_threads} \
full \

./ResHist_Mfem.sh \
async_afacx \
${smoother} \
${num_threads} \
semi \

./ResHist_Mfem.sh \
afacx \
${smoother} \
${num_threads} \
full \

./ResHist_Mfem.sh \
mult \
${smoother} \
${num_threads} \
full \
