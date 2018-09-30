#!/bin/bash

problem="27pt"
num_smooth_sweeps=1
smoother="L1j"
n=20

t_list=(1 2 4 8 16 32)

for t in ${t_list[@]}
do
	num_threads=${t}
	./Threads_ResHist.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${n}
done
