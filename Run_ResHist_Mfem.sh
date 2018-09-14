#!/bin/bash

./ResHist_Mfem.sh \
async_multadd \
symm_j \
32 \
full \

./ResHist_Mfem.sh \
async_multadd \
symm_j \
32 \
semi \

./ResHist_Mfem.sh \
async_multadd \
semi_async_gs \
32 \
full \

./ResHist_Mfem.sh \
async_multadd \
semi_async_gs \
32 \
semi \

./ResHist_Mfem.sh \
async_afacx \
semi_async_gs \
32 \
full \

./ResHist_Mfem.sh \
async_afacx \
semi_async_gs \
32 \
semi \

./ResHist_Mfem.sh \
afacx \
semi_async_gs \
32 \
full \

./ResHist_Mfem.sh \
mult \
semi_async_gs \
32 \
full \

./ResHist_Mfem.sh \
hybrid_jgs \
semi_async_gs \
32 \
full \
