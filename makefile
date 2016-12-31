#standard places to find cuda files
CUDA_INC = /usr/local/cuda/includes
CUDA_LIB = /usr/local/cuda/lib64
CUDA_LIB2 = /usr/local/cuda/lib
#CUDA_INC = /usr/local/cuda-8.0/includes
#CUDA_LIB = /usr/local/cuda-8.0/lib64
#CUDA_LIB2 = /usr/local/cuda-8.0/lib

CXX := g++
CC := gcc
LINK := g++ #-fPIC
NVCC := nvcc

INCLUDES = -I. -I./src/ -I./ext_src/ -I./inc/ -I$(CUDA_INC) -I/home/user/CGAL/CGAL-4.9/include -I/opt/local/include
LIB_CUDA = -L. -L$(CUDA_LIB) -L$(CUDA_LIB2) -lcuda -lcudart
LIB_CGAL = -L/home/user/CGAL/CGAL-4.9/lib -lCGAL -lCGAL_Core -lgmp -lmpfr
LIB_NETCDF = -lnetcdf_c++ -lnetcdf -L/opt/local/lib

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

debug: CXXFLAGS += -g
debug: NVCCFLAGS += -g -lineinfo -Xptxas --generate-line-info
debug: build

PROGS= delGPU.out

build: $(PROGS)

PROG_OBJS= obj/runellipse.o obj/voroguppy.o obj/runplates.o obj/runMakeDatabase.o

CLASS_OBJS= obj/DelaunayLoc.o obj/Delaunay1.o obj/DelaunayCGAL.o obj/gpucell.o obj/DelaunayMD.o obj/spv2d.o obj/hilbert_curve.o obj/avm2d.o

CUOBJS= obj/cuobj/gpucell.cu.o obj/cuobj/DelaunayMD.cu.o obj/cuobj/spv2d.cu.o obj/cuobj/avm2d.cu.o

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

#obj/runellipse.o:runellipse.cpp
#	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<

#obj/runplates.o:runplates.cpp
#	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<

#obj/runMakeDatabase.o:runMakeDatabase.cpp
#	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<


delGPU.out: obj/voroguppy.o $(CLASS_OBJS) $(CUOBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+


run: build
	./delGPU.out

clean:
	rm -f $(PROG_OBJS) $(CLASS_OBJS) $(CUOBJS) delGPU.out

