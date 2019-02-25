SRC_DIR=src/

SMEM_CPP_FILES = $(SRC_DIR)Laplacian.cpp \
	   	 $(SRC_DIR)Elasticity.cpp \
	   	 $(SRC_DIR)Maxwell.cpp \
	   	 $(SRC_DIR)Misc.cpp \
	   	 $(SRC_DIR)SEQ_MatVec.cpp \
           	 $(SRC_DIR)SEQ_Smooth.cpp \
           	 $(SRC_DIR)SEQ_AMG.cpp \
	   	 $(SRC_DIR)SMEM_Setup.cpp \
           	 $(SRC_DIR)SMEM_MatVec.cpp \
           	 $(SRC_DIR)SMEM_Smooth.cpp \
           	 $(SRC_DIR)SMEM_Sync_AMG.cpp \
           	 $(SRC_DIR)SMEM_Async_AMG.cpp \
           	 $(SRC_DIR)SMEM_Solve.cpp \

DMEM_CPP_FILES = $(SRC_DIR)Misc.cpp \
		 $(SRC_DIR)DMEM_Laplacian.cpp \
		 $(SRC_DIR)DMEM_Setup.cpp \
		 $(SRC_DIR)DMEM_Comm.cpp \
		 $(SRC_DIR)DMEM_MatVec.cpp \
		 $(SRC_DIR)DMEM_Async_AMG.cpp \

CPP_COMPILE = g++ -fopenmp -O3
ICPC_COMPILE = icpc -qopenmp -std=c++11 -O3 -Wall
MPIICPC_COMPILE = mpiicpc -qopenmp -std=c++11 -O3 -mkl #-g -w3
NERSC_COMPILE = CC -qopenmp -std=c++11 -O3 -mkl
MPICXX_COMPILE =  mpicxx -fopenmp -g -std=c++0x -O3 #-Wall

NERSC_INCLUDE = -I/global/homes/j/jwolfson/async-multigrid/src/hypre_include -I/global/homes/j/jwolfson/async-multigrid/mfem
GOTHAM_INCLUDE = -I/home/jwp3local/async-multigrid/src/hypre_include -I/home/jwp3local/async-multigrid/mfem

INCLUDE=$(GOTHAM_INCLUDE)
COMPILE=$(MPIICPC_COMPILE)
LIBS = libHYPRE.a libmfem.a
VARS = -DEIGEN_DONT_VECTORIZE=1

all: clean DMEM_Main

SMEM_Main: $(SRC_DIR)SMEM_Main.cpp
	$(COMPILE) $(SRC_DIR)SMEM_Main.cpp $(SMEM_CPP_FILES) $(VARS) $(LIBS) $(INCLUDE) -o SMEM_Main
	cp SMEM_Main experiments/SMEM_Main

DMEM_Main: $(SRC_DIR)DMEM_Main.cpp
	$(COMPILE) $(SRC_DIR)DMEM_Main.cpp $(DMEM_CPP_FILES) $(VARS) $(LIBS) $(INCLUDE) -o DMEM_Main

TextToBin: $(SRC_DIR)TextToBin.cpp
	$(ICPC_COMPILE) $(SRC_DIR)TextToBin.cpp -o TextToBin

clean:
	rm -f SMEM_Main DMEM_Main
