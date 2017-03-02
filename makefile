#standard places to find cuda files
CUDA_INC = /usr/local/cuda/includes
CUDA_LIB = /usr/local/cuda/lib64
CUDA_LIB2 = /usr/local/cuda/lib

CXX := g++
CC := gcc
LINK := g++ #-fPIC
NVCC := nvcc

INCLUDES = -I. -I./src/ -I./ext_src/ -I./inc/ -I$(CUDA_INC) -I/home/user/CGAL/CGAL-4.9/include -I/opt/local/include
INCLUDES += -I/usr/local/Cellar/cgal/4.9/include -I/usr/local/Cellar/boost/1.62.0/include -I/usr/local/Cellar/gmp/6.1.2/include -I/usr/local/Cellar/mpfr/3.1.5/include
LIB_CUDA = -L. -L$(CUDA_LIB) -L$(CUDA_LIB2) -lcuda -lcudart
LIB_CGAL += -L/usr/local/Cellar/cgal/4.9/lib -L/usr/local/Cellar/gmp/6.1.2/lib -L/usr/local/Cellar/mpfr/3.1.5/lib
LIB_CGAL += -L/home/user/CGAL/CGAL-4.9/lib -lCGAL -lCGAL_Core -lgmp -lmpfr
LIB_NETCDF = -lnetcdf -lnetcdf_c++ -L/opt/local/lib

#common flags
COMMONFLAGS += $(INCLUDES) -std=c++11 -DCGAL_DISABLE_ROUNDING_MATH_CHECK -O3
NVCCFLAGS += -arch=sm_35 -D_FORCE_INLINES $(COMMONFLAGS) -Wno-deprecated-gpu-targets #-Xptxas -fmad=false#-O0#-dlcm=ca#-G
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -w -frounding-math
CFLAGS += $(COMMONFLAGS) -frounding-math

CUOBJ_DIR=obj/cuobj
OBJ_DIR=obj
SRC_DIR=src
#target rules
all:build

float: CXXFLAGS += -DSCALARFLOAT
float: NVCCFLAGS += -DSCALARFLOAT
float: build

debug: CXXFLAGS += -g -DCUDATHREADSYNC
debug: NVCCFLAGS += -g -lineinfo -Xptxas --generate-line-info # -G
debug: build

PROGS= avmGPU.out spvMSD.out Minimize.out spvGPU.out

build: $(PROGS)

PROG_OBJS=obj/activeVertex.o obj/voronoi.o obj/runMakeDatabase.o obj/minimize.o

CLASS_OBJS= obj/DelaunayLoc.o obj/Delaunay1.o obj/DelaunayCGAL.o obj/cellListGPU.o obj/DelaunayMD.o obj/hilbert_curve.o obj/EnergyMinimizerFIRE2D.o
CLASS_OBJS+=obj/Simple2DCell.o obj/Simple2DActiveCell.o
CLASS_OBJS+=obj/selfPropelledParticleDynamics.o obj/selfPropelledCellVertexDynamics.o obj/brownianParticleDynamics.o
CLASS_OBJS+=obj/avm2d.o obj/spv2d.o

CUOBJS= obj/cuobj/cellListGPU.cu.o obj/cuobj/DelaunayMD.cu.o
CUOBJS+=obj/cuobj/Simple2DCell.cu.o
CUOBJS+=obj/cuobj/EnergyMinimizerFIRE2D.cu.o
CUOBJS+=obj/cuobj/simpleEquationOfMotion.cu.o obj/cuobj/selfPropelledParticleDynamics.cu.o obj/cuobj/selfPropelledCellVertexDynamics.cu.o obj/cuobj/brownianParticleDynamics.cu.o
CUOBJS+=obj/cuobj/spv2d.cu.o obj/cuobj/avm2d.cu.o
#cuda objects
$(CUOBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA)  -o $@ -c $<

#cpp class objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#program objects
$(OBJ_DIR)/%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<


###
#Programs
##

avmGPU.out: obj/activeVertex.o $(CLASS_OBJS) $(CUOBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+
spvGPU.out: obj/voronoi.o $(CLASS_OBJS) $(CUOBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+
Minimize.out: obj/minimize.o $(CLASS_OBJS) $(CUOBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+
spvMSD.out: obj/runMakeDatabase.o $(CLASS_OBJS) $(CUOBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+

run: build
	./spvGPU.out
	./spvMSD.out
	./avmGPU.out
	./Minimize.out

clean:
	rm -f $(PROG_OBJS) $(CLASS_OBJS) $(CUOBJS) spvGPU.out avmGPU.out

