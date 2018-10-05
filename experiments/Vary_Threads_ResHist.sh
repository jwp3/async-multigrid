#!/bin/bash

problem="7pt"
num_smooth_sweeps=1
n=20

smoother="L1j"
t_list=(1 2 4 8 16 32)

for t in ${t_list[@]}
do
	num_threads=${t}
	./Threads_ResHist.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${n}
done


smoother="async_gs"
t_list=(1 2 4 8 16 32)

for t in ${t_list[@]}
do
        num_threads=${t}
        ./Threads_ResHist.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${n}
done
