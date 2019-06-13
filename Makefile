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
		 $(SRC_DIR)DMEM_Misc.cpp \
		 $(SRC_DIR)DMEM_Laplacian.cpp \
		 $(SRC_DIR)DMEM_ParMfem.cpp \
		 $(SRC_DIR)DMEM_Setup.cpp \
		 $(SRC_DIR)DMEM_Comm.cpp \
		 $(SRC_DIR)DMEM_MatVec.cpp \
		 $(SRC_DIR)DMEM_Add.cpp \
		 $(SRC_DIR)DMEM_Mult.cpp \
		 $(SRC_DIR)DMEM_Test.cpp \

CPP_COMPILE = g++ -fopenmp -O3
ICPC_COMPILE = icpc -qopenmp -std=c++11 -O3 -Wall
MPIICPC_COMPILE = mpiicpc -qopenmp -std=c++11 -O3 -mkl #-g -w3
MPICXX_COMPILE =  mpicxx -mkl -fopenmp -std=c++0x -O3 #-Wall

NERSC_CORI_COMPILE = CC -qopenmp -std=c++11 -O3 -mkl
LLNL_QUARTZ_COMPILE = $(MPICXX_COMPILE)
GOTHAM_COMPILE = $(MPIICPC_COMPILE)

NERSC_CORI_INCLUDE = -I/global/homes/j/jwolfson/async-multigrid/src/hypre_include -I/global/homes/j/jwolfson/async-multigrid/mfem
GOTHAM_INCLUDE = -I/home/jwp3local/async-multigrid/mfem/mfem-3.4 -I/home/jwp3local/async-multigrid/mfem/hypre-2.11.2/src/hypre/include
LLNL_QUARTZ_INCLUDE = -I/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem/mfem-3.4 -I/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem/hypre/src/hypre/include

INCLUDE=$(LLNL_QUARTZ_INCLUDE)
COMPILE=$(LLNL_QUARTZ_COMPILE)
LIBS = ./mfem/mfem-3.4/libmfem.a ./mfem/hypre/src/lib/libHYPRE.a ./mfem/metis-5.1.0/libmetis.a

all: clean DMEM_Main

SMEM_Main: $(SRC_DIR)SMEM_Main.cpp
	$(COMPILE) $(SRC_DIR)SMEM_Main.cpp $(SMEM_CPP_FILES) -DEIGEN_DONT_VECTORIZE=1 $(LIBS) $(INCLUDE) -o SMEM_Main
	cp SMEM_Main experiments/SMEM_Main

DMEM_Main: $(SRC_DIR)DMEM_Main.cpp
	$(COMPILE) $(SRC_DIR)DMEM_Main.cpp $(DMEM_CPP_FILES) $(VARS) $(LIBS) $(INCLUDE) -o DMEM_Main

TextToBin: $(SRC_DIR)TextToBin.cpp
	$(ICPC_COMPILE) $(SRC_DIR)TextToBin.cpp -o TextToBin

clean:
	rm -f SMEM_Main DMEM_Main
