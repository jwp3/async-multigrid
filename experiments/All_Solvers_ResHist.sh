#!/bin/bash

num_threads=32
problem="7pt"
num_smooth_sweeps=1
n=20

smoother="L1j"
./Solvers_ResHist.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${n} 
smoother="j"
./Solvers_ResHist.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${n}
smoother="hybrid_jgs"
./Solvers_ResHist.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${n}
smoother="async_gs"
./Solvers_ResHist.sh ${smoother} ${num_threads} ${problem} ${num_smooth_sweeps} ${n}
