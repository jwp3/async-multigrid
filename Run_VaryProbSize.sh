#!/bin/bash

num_threads=272

./VaryProbSize.sh \
async_multadd \
j \
5pt \
${num_threads} \
full \

./VaryProbSize.sh \
async_afacx \
j \
5pt \
${num_threads} \
full \

./VaryProbSize.sh \
async_multadd \
j \
5pt \
${num_threads} \
semi \

./VaryProbSize.sh \
async_afacx \
j \
5pt \
${num_threads} \
semi \

./VaryProbSize.sh \
mult \
j \
5pt \
${num_threads} \
full \

./VaryProbSize.sh \
afacx \
j \
5pt \
${num_threads} \
full \
