#include ./mfem_lassen/hypre_amg-setup/src/config/Makefile.config
#
#CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
#
#CDEFS = -DHYPRE_TIMING -DHYPRE_FORTRAN
#
#C_COMPILE_FLAGS = \
# -I$(srcdir)\
# -I$(srcdir)/..\
# -I${HYPRE_BUILD_DIR}/include\
# ${CINCLUDES}\
# ${CDEFS}
#
#MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS}
#LIBFLAGS = ${LDFLAGS} ${LIBS}
#
#XLINK = -Xlinker=-rpath,${HYPRE_BUILD_DIR}/lib
#
#LFLAGS =\
# -L${HYPRE_BUILD_DIR}/lib -lHYPRE\
# ${XLINK}\
# ${MPILIBFLAGS}\
# ${LAPACKLIBFLAGS}\
# ${BLASLIBFLAGS}\
# ${LIBFLAGS}
#
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
		 $(SRC_DIR)DMEM_BuildMatrix.cpp \
		 $(SRC_DIR)DMEM_ParMfem.cpp \
		 $(SRC_DIR)DMEM_Setup.cpp \
		 $(SRC_DIR)DMEM_Comm.cpp \
		 $(SRC_DIR)DMEM_MatVec.cpp \
		 $(SRC_DIR)DMEM_Add.cpp \
		 $(SRC_DIR)DMEM_Mult.cpp \
		 $(SRC_DIR)DMEM_Test.cpp \
		 $(SRC_DIR)DMEM_Smooth.cpp \

#nvcc -O2 -ccbin=mpixlC -gencode arch=compute_60,"code=sm_60" -expt-extended-lambda -dc --std=c++11 -Xcompiler -Wno-deprecated-register --x cu -Xcompiler "-O2 " -DHAVE_CONFIG_H

CPP_COMPILE = g++ -fopenmp -O3
ICPC_COMPILE = icpc -qopenmp -std=c++11 -O3 -Wall
MPIICPC_COMPILE = mpiicpc -qopenmp -std=c++11 -O3 -mkl #-g -w3
MPICXX_COMPILE =  mpicxx -fopenmp -std=c++0x -O3 #-Wall

COMPILE_CORI = CC -qopenmp -std=c++11 -O3 -mkl
COMPILE_QUARTZ = $(MPICXX_COMPILE)
COMPILE_LASSEN = nvcc -O2 -ccbin=mpixlC -gencode arch=compute_70,"code=sm_70" -expt-extended-lambda --std=c++11 -Xcompiler -Wno-deprecated-register -Xcompiler "-O2 " -DHAVE_CONFIG_H -lm -lcusparse -lcudart -lcublas -lnvToolsExt
#COMPILE_LASSEN = nvcc -O2 -ccbin=mpixlC -gencode arch=compute_70,"code=sm_70" -expt-extended-lambda -std=c++11 -Xcompiler -Wno-deprecated-register -Xcompiler "-O2 " -lm -DHAVE_CONFIG_H -L/usr/tce/packages/cuda/cuda-10.1.243/lib64 -lcurand -lcusparse -lcudart -lcublas -lnvToolsExt #mpicxx -std=c++0x -L/usr/tce/packages//cuda-10.1.243/lib64 -lcudart #nvcc -ccbin=mpicxx --std=c++11
COMPILE_GOTHAM = $(MPIICPC_COMPILE)

INCLUDE_CORI = -I/global/homes/j/jwolfson/async-multigrid/src/hypre_include -I/global/homes/j/jwolfson/async-multigrid/mfem
INCLUDE_GOTHAM = -I/home/jwp3local/async-multigrid/mfem/mfem-3.4 -I/home/jwp3local/async-multigrid/mfem/hypre-2.11.2/src/hypre/include

INCLUDE_QUARTZ = -I/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem_quartz/mfem-3.4 -I/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem_quartz/hypre/src/hypre/include
INCLUDE_LASSEN =\
 -I/usr/tce/packages/cuda/cuda-10.1.243/include\
 -I/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem_lassen/mfem-3.4\
 -I/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem_lassen/hypre_amg-setup/src/hypre/include\

INCLUDE=$(INCLUDE_QUARTZ)
COMPILE=$(COMPILE_QUARTZ)

LIBS_QUARTZ = ./mfem_quartz/mfem-3.4/libmfem.a ./mfem_quartz/hypre/src/lib/libHYPRE.a ./mfem_quartz/metis-5.1.0/libmetis.a
LIBS_LASSEN =\
 -L/usr/tce/packages/cuda/cuda-10.1.243/lib64\
 -L/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem_lassen/hypre_amg-setup/src/hypre/lib\
 -L/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem_lassen/metis-5.1.0\
 -L/g/g13/wolfsonp/Summer2019/AsyncMultigrid/async-multigrid/mfem_lassen/mfem-3.4/\
 -lmfem -lmetis -lHYPRE -lcusparse -lcurand -lcudart\

DMEM_lassen: clean_lassen DMEM_Main_lassen

DMEM_quartz: clean_quartz DMEM_Main_quartz

SMEM_Main: $(SRC_DIR)SMEM_Main.cpp
	$(COMPILE) $(SRC_DIR)SMEM_Main.cpp $(SMEM_CPP_FILES) -DEIGEN_DONT_VECTORIZE=1 $(LIBS) $(INCLUDE) -o SMEM_Main
	cp SMEM_Main experiments/SMEM_Main

DMEM_Main_lassen: $(SRC_DIR)DMEM_Main.cpp
	nvcc -O2 -ccbin=mpixlC -gencode arch=compute_70,"code=sm_70" -std=c++11 $(LIBS_LASSEN) $(INCLUDE_LASSEN) --x cu $(DMEM_CPP_FILES) $(SRC_DIR)DMEM_Main.cpp -o DMEM_Main_lassen

DMEM_Main_quartz: $(SRC_DIR)DMEM_Main.cpp
	$(COMPILE_QUARTZ) $(SRC_DIR)DMEM_Main.cpp $(DMEM_CPP_FILES) $(VARS) $(LIBS_QUARTZ) $(INCLUDE_QUARTZ) -o DMEM_Main_quartz

TextToBin: $(SRC_DIR)TextToBin.cpp
	$(ICPC_COMPILE) $(SRC_DIR)TextToBin.cpp -o TextToBin

clean_lassen:
	rm -f DMEM_Main_lassen

clean_quartz:
	rm -f DMEM_Main_quartz
