SRC_DIR=src/

CPP_FILES = $(SRC_DIR)Laplacian.cpp \
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

CPP_COMPILE = g++ -fopenmp -O3
ICPC_COMPILE = icpc -qopenmp -std=c++11 -O3 -Wall
MPIICPC_COMPILE = mpiicpc -qopenmp -std=c++11 -O3 -mkl #-g #-w3
NERSC_COMPILE = CC -qopenmp -std=c++11 -O3 -mkl
MPICXX_COMPILE =  mpicxx -fopenmp -g -std=c++0x -O3 -Wall

COMPILE=$(MPIICPC_COMPILE)

all: clean MAIN

DMEM_LIBS =
LIBS = libHYPRE.a libmfem.a

NERSC_INCLUDE = -I/global/homes/j/jwolfson/async-multigrid/src/hypre_include -I/global/homes/j/jwolfson/async-multigrid/mfem
GOTHAM_INCLUDE = -I/home/jwp3local/async-multigrid/src/hypre_include -I/home/jwp3local/async-multigrid/mfem

INCLUDE=$(GOTHAM_INCLUDE)

MAIN: $(SRC_DIR)Main.cpp
	$(COMPILE) $(SRC_DIR)Main.cpp $(CPP_FILES) $(LIBS) $(INCLUDE) -o Main

DMEM: $(SRC_DIR)DMEM_Main.cpp

TextToBin: $(SRC_DIR)TextToBin.cpp
	$(ICPC_COMPILE) $(SRC_DIR)TextToBin.cpp -o TextToBin

clean:
	rm -f Main
