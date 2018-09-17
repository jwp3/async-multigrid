#!/bin/bash
srun -n1 -u ./Main \
	-num_cycles 20 \
	-num_threads 272 \
	-solver afacx \
	-problem 5pt \
	-n 1000 \
	-async_type semi \
	-converge_test_type all \
	-smoother j \
	-num_smooth_sweeps 3 \
	-smooth_weight .8 \
	-mfem_ref_levels 4 \
	-mfem_mesh_file "./mfem/data/ball-nurbs.mesh" \
	-agg_nl 0 \
