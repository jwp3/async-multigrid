#!/bin/bash

num_threads=272
smoother="j"

./VaryProbSize.sh \
async_multadd \
${smoother} \
5pt \
${num_threads} \
full \

./VaryProbSize.sh \
async_multadd \
${smoother} \
5pt \
${num_threads} \
semi \

./VaryProbSize.sh \
async_afacx \
${smoother} \
5pt \
${num_threads} \
full \

./VaryProbSize.sh \
async_afacx \
${smoother} \
5pt \
${num_threads} \
semi \

./VaryProbSize.sh \
mult \
${smoother} \
5pt \
${num_threads} \
full \

./VaryProbSize.sh \
multadd \
${smoother} \
5pt \
${num_threads} \
full \

./VaryProbSize.sh \
afacx \
${smoother} \
5pt \
${num_threads} \
full \
