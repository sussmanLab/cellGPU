#This makefile has crazy-lloking paths and includes, reflecting the different operating systems and computers regulating used by the developer. For your system you can definitely, definitely simplify this. Currently, this will auto-compile all cpp files in the main directory, and make objects for all cpp and cu files in the src directory

#standard places to find cuda files
CUDA_INC = /usr/local/cuda/includes
CUDA_LIB = /usr/local/cuda/lib64
CUDA_LIB2 = /usr/local/cuda/lib

CXX := g++
CC := gcc
LINK := g++ #-fPIC
NVCC := nvcc

INCLUDES = -I. -I./src/ -I./ext_src/ -I./inc/ -I$(CUDA_INC) -I/home/user/CGAL/CGAL-4.9/include -I/opt/local/include -I/usr/local/include/eigen3 -I/usr/include/eigen3
INCLUDES += -I/usr/local/Cellar/cgal/4.9/include -I/usr/local/Cellar/boost/1.62.0/include -I/usr/local/Cellar/gmp/6.1.2/include -I/usr/local/Cellar/mpfr/3.1.5/include -I/usr/local/Cellar/netcdf/4.4.1.1_4/include -I/home/user/netcdf-cxx/include
LIB_CUDA = -L. -L$(CUDA_LIB) -L$(CUDA_LIB2) -lcuda -lcudart
LIB_CGAL += -L/usr/local/Cellar/cgal/4.9/lib -L/usr/local/Cellar/gmp/6.1.2/lib -L/usr/local/Cellar/mpfr/3.1.5/lib
LIB_CGAL += -L/home/user/CGAL/CGAL-4.9/lib -lCGAL -lCGAL_Core -lgmp -lmpfr
LIB_NETCDF = -lnetcdf -lnetcdf_c++ -L/opt/local/lib -L/usr/local/Cellar/netcdf/4.4.1.1_4/lib -L//home/user/netcdf-cxx/lib

#common flags
COMMONFLAGS += $(INCLUDES) -std=c++11 -DCGAL_DISABLE_ROUNDING_MATH_CHECK -O3 -D_FORCE_INLINES
NVCCFLAGS += -arch=sm_35 $(COMMONFLAGS) -Wno-deprecated-gpu-targets #-Xptxas -fmad=false#-O0#-dlcm=ca#-G
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -w -frounding-math
CFLAGS += $(COMMONFLAGS) -frounding-math

CUOBJ_DIR=obj/cuobj
OBJ_DIR=obj
SRC_DIR=src
BIN_DIR=.

.SECONDARY:

PROGS := $(wildcard *.cpp)
PROG_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.main.o,$(PROGS))
PROG_MAINS := $(patsubst %.cpp,$(BIN_DIR)/%.out,$(PROGS))


CPP_FILES := $(wildcard src/*.cpp)
CLASS_OBJS = $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
CU_FILES := $(wildcard src/*.cu)
CUOBJS = $(patsubst src/%.cu,$(CUOBJ_DIR)/%.cu.o,$(CU_FILES))


#cuda objects
$(CUOBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA)  -o $@ -c $<

#cpp class objects
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#program objects
$(OBJ_DIR)/%.main.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#Programs
%.out: $(OBJ_DIR)/%.main.o $(CLASS_OBJS) $(CUOBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+

#target rules

all:build

float: CXXFLAGS += -DSCALARFLOAT
float: NVCCFLAGS += -DSCALARFLOAT
float: build

debug: CXXFLAGS += -g -DCUDATHREADSYNC
debug: NVCCFLAGS += -g -lineinfo -Xptxas --generate-line-info # -G
debug: build
build:$(PROG_MAINS)

clean: 
	rm -f $(PROG_OBJS) $(CLASS_OBJS) $(CUOBJS) $(PROG_OBJS) $(PROG_MAINS)
