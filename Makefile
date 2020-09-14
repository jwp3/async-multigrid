SRC_DIR=src/
OBJ_DIR=obj/

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
		 $(SRC_DIR)SMEM_ExtendedSystem.cpp \
		 $(SRC_DIR)SMEM_Cheby.cpp \

SMEM_HPP_FILES = $(SRC_DIR)Main.hpp \
		 $(SRC_DIR)Laplacian.hpp \
                 $(SRC_DIR)Elasticity.hpp \
                 $(SRC_DIR)Maxwell.hpp \
                 $(SRC_DIR)Misc.hpp \
                 $(SRC_DIR)SEQ_MatVec.hpp \
                 $(SRC_DIR)SEQ_Smooth.hpp \
                 $(SRC_DIR)SEQ_AMG.hpp \
                 $(SRC_DIR)SMEM_Setup.hpp \
                 $(SRC_DIR)SMEM_MatVec.hpp \
                 $(SRC_DIR)SMEM_Smooth.hpp \
                 $(SRC_DIR)SMEM_Sync_AMG.hpp \
                 $(SRC_DIR)SMEM_Async_AMG.hpp \
                 $(SRC_DIR)SMEM_Solve.hpp \
                 $(SRC_DIR)SMEM_ExtendedSystem.hpp \
                 $(SRC_DIR)SMEM_Cheby.hpp \

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
		 $(SRC_DIR)DMEM_Eig.cpp \

SMEM_OBJ = $(SMEM_CPP_FILES:$(SRC_DIR)%.cpp=$(OBJ_DIR)%.o) obj/SMEM_Main.o

CPP_COMPILE = g++ -fopenmp -O3 -std=c++0x -g
ICPC_COMPILE = icpc -qopenmp -std=c++11 -O3 -Wall
MPIICPC_COMPILE = mpiicpc -qopenmp -std=c++11 -O3 #-mkl -g -w3
MPICXX_COMPILE =  mpicxx -g -fopenmp -std=c++0x -O3 #-Wall

COMPILE_CORI = CC -qopenmp -std=c++11 -O3 -mkl
COMPILE_QUARTZ =  mpicxx -fopenmp -std=c++0x -O3
COMPILE_LASSEN = nvcc -g -O3 -std=c++11 -x=cu --expt-extended-lambda -arch=sm_70 -ccbin mpicxx
COMPILE_JORDI = $(CPP_COMPILE) -D WINDOWS

INCLUDE_CORI = -I/global/homes/j/jwolfson/async-multigrid/src/hypre_include -I/global/homes/j/jwolfson/async-multigrid/mfem
INCLUDE_GOTHAM = -I/home/jwp3local/async-multigrid/mfem/mfem-3.4 -I/home/jwp3local/async-multigrid/mfem/hypre-2.11.2/src/hypre/include

INCLUDE_JORDI = \
 -I./mfem/hypre/src/hypre/include \
 -I./mfem/metis-5.1.0/include \
 -I./eigen/Eigen \
 -I/usr/include \

INCLUDE_QUARTZ = \
 -I./mfem_quartz/mfem-4.0 \
 -I./mfem_quartz/hypre/src/hypre/include \
 -I./mfem_quartz/metis-5.1.0/include \
 -I./mfem_quartz/eigen/Eigen \

INCLUDE_LASSEN = \
 -I/usr/tce/packages/cuda/cuda-10.1.243/include \
 -I./mfem_lassen/hypre/src/hypre/include \
 -I./mfem_lassen/mfem-4.0 \
 -I./mfem_lassen/metis-5.1.0/include \

LIBS_JORDI = \
 ./mfem/hypre/src/lib/libHYPRE.a \
 ./mfem/metis-5.1.0/lib/libmetis.a \
 /usr/lib/libmpi.dll.a \

LIBS_QUARTZ = \
 ./mfem_quartz/mfem-4.0/libmfem.a \
 ./mfem_quartz/hypre/src/lib/libHYPRE.a \
 ./mfem_quartz/metis-5.1.0/lib/libmetis.a \

LIBS_LASSEN = \
 -L./mfem_lassen/mfem-4.0 \
 -lmfem \
 -L./mfem_lassen/hypre/src/hypre/lib \
 -lHYPRE \
 -L./mfem_lassen/metis-5.1.0/lib \
 -lmetis \
 -lrt \
 -lcusparse \
 -lcurand \
 -lcudart \

COMPILE = $(COMPILE_JORDI)
LIBS = $(LIBS_JORDI)
INCLUDE = $(INCLUDE_JORDI)

SMEM_jordi: SMEM_Main_jordi

SMEM_lassen: clean_smem_lassen SMEM_Main_lassen

SMEM_quartz: clean_smem_quartz SMEM_Main_quartz

SMEM_quartz2: clean_smem_quartz2 SMEM_Main_quartz2

DMEM_lassen: clean_lassen DMEM_Main_lassen

DMEM_lassen2: clean_lassen2 DMEM_Main_lassen2

DMEM_quartz: clean_quartz DMEM_Main_quartz2

SMEM_Main_jordi: $(SMEM_OBJ)
	$(COMPILE) $(INCLUDE) $^ $(LIBS) -o $@ 

#SMEM_Main_jordi: $(SRC_DIR)SMEM_Main.cpp
#	$(COMPILE_JORDI) -c $(SMEM_CPP_FILES) $(SRC_DIR)SMEM_Main.cpp $(INCLUDE_JORDI)
#	mkdir -p build
#	mv $(subst src, .,$(SMEM_CPP_FILES:.cpp=.o)) build/
#	mv SMEM_Main.o build/ 
#	$(COMPILE_JORDI) $(SRC_DIR)SMEM_Main.cpp $(LIBS_JORDI) $(INCLUDE_JORDI) -o SMEM_Main_jordi

SMEM_Main_quartz: $(SRC_DIR)SMEM_Main.cpp
	$(COMPILE_QUARTZ) $(SRC_DIR)SMEM_Main.cpp $(SMEM_CPP_FILES) $(LIBS_QUARTZ) $(INCLUDE_QUARTZ) -o SMEM_Main_quartz

SMEM_Main_lassen: $(SRC_DIR)SMEM_Main.cpp
	$(COMPILE_LASSEN) $(SRC_DIR)SMEM_Main.cpp $(SMEM_CPP_FILES) $(LIBS_LASSEN) $(INCLUDE_LASSEN) -o SMEM_Main_lassen

SMEM_Main_quartz2: $(SRC_DIR)SMEM_Main.cpp
	$(COMPILE_QUARTZ) $(SRC_DIR)SMEM_Main.cpp $(SMEM_CPP_FILES) $(LIBS_QUARTZ) $(INCLUDE_QUARTZ) -o SMEM_Main_quartz2

DMEM_Main_lassen: $(SRC_DIR)DMEM_Main.cpp
	$(COMPILE_LASSEN) $(INCLUDE_LASSEN) $(DMEM_CPP_FILES) $(SRC_DIR)DMEM_Main.cpp -o DMEM_Main_lassen $(LIBS_LASSEN)

DMEM_Main_lassen2: $(SRC_DIR)DMEM_Main.cpp
	$(COMPILE_LASSEN) $(INCLUDE_LASSEN) $(DMEM_CPP_FILES) $(SRC_DIR)DMEM_Main.cpp -o DMEM_Main_lassen2 $(LIBS_LASSEN)

DMEM_Main_quartz2: $(SRC_DIR)DMEM_Main.cpp
	$(COMPILE_QUARTZ) $(SRC_DIR)DMEM_Main.cpp $(DMEM_CPP_FILES) $(LIBS_QUARTZ) $(INCLUDE_QUARTZ) -o DMEM_Main_quartz2

TextToBin: $(SRC_DIR)TextToBin.cpp
	$(COMPILE_QUARTZ) $(SRC_DIR)TextToBin.cpp -o TextToBin

clean_smem_jordi:
	rm -f SMEM_Main_jordi

clean_lassen:
	rm -f DMEM_Main_lassen

clean_lassen2:
	rm -f DMEM_Main_lassen2

clean_quartz:
	rm -f DMEM_Main_quartz2

clean_smem_lassen:
	rm -f SMEM_Main_lassen

clean_smem_quartz:
	rm -f SMEM_Main_quartz

clean_smem_quartz2:
	rm -f SMEM_Main_quartz2

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp | $(OBJ_DIR)
	$(COMPILE) $(INCLUDE) -c $< -o $@
