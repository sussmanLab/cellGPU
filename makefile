#This makefile has crazy-looking paths and includes, reflecting a makefile that works on the different
#operating systems and computers regulating used by the developer. For your system you can definitely,
#definitely simplify this. Currently, this will auto-compile all cpp files in the main directory,
#and make objects for all cpp and cu files in the src directory

CXX := g++
CC := gcc
LINK := g++ #-fPIC
NVCC := nvcc

INCLUDES := -I. -I./inc/ -I/opt/local/include
INCLUDES += -I/usr/local/cuda/includes -I/usr/local/cuda/include
INCLUDES += -I/home/user/CGAL/CGAL-4.9/include -I/usr/local/Cellar/cgal/4.9/include -I/usr/local/Cellar/gmp/6.1.2/include -I/usr/local/Cellar/mpfr/3.1.5/include -I/usr/local/Cellar/boost/1.62.0/include
INCLUDES += -I/usr/local/Cellar/netcdf/4.4.1.1_4/include -I/home/user/netcdf-cxx/include -I/usr/local/include/eigen3 -I/usr/include/eigen3

LIB_CUDA := -L. -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcuda -lcudart
LIB_CGAL := -L/usr/local/Cellar/cgal/4.9/lib -L/usr/local/Cellar/gmp/6.1.2/lib -L/usr/local/Cellar/mpfr/3.1.5/lib  -L/home/user/CGAL/CGAL-4.9/lib -lCGAL -lCGAL_Core -lgmp -lmpfr
LIB_NETCDF = -lnetcdf -lnetcdf_c++ -L/opt/local/lib -L/usr/local/Cellar/netcdf/4.4.1.1_4/lib -L/home/user/netcdf-cxx/lib

#common flags
COMMONFLAGS += -std=c++11 -DCGAL_DISABLE_ROUNDING_MATH_CHECK -O3 -D_FORCE_INLINES
NVCCFLAGS += -arch=sm_35 $(COMMONFLAGS) -Wno-deprecated-gpu-targets #-Xptxas -fmad=false#-O0#-dlcm=ca#-G
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -w -frounding-math
CFLAGS += $(COMMONFLAGS) -frounding-math

CUOBJ_DIR=obj/cuobj
MODULES = databases models updaters utility
INCLUDES += -I./inc/databases -I./inc/models -I./inc/updaters -I./inc/utility
OBJ_DIR=obj
SRC_DIR=src
BIN_DIR=.

.SECONDARY:

PROGS := $(wildcard *.cpp)
PROG_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.main.o,$(PROGS))
PROG_MAINS := $(patsubst %.cpp,$(BIN_DIR)/%.out,$(PROGS))


CPP_FILES := $(wildcard src/*/*.cpp)
CPP_FILES += $(wildcard src/*.cpp)
CU_FILES := $(wildcard src/*/*.cu)
CU_FILES += $(wildcard src/*.cu)

CLASS_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_FILES))


#cuda objects
$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/%.cu 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA)  -o $@ -c $<

#cpp class objects
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#program objects
$(OBJ_DIR)/%.main.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) $(LIB_CGAL) -o $@ -c $<

#Programs
%.out: $(OBJ_DIR)/%.main.o $(CLASS_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+

#target rules

all:build

float: CXXFLAGS += -DSCALARFLOAT
float: NVCCFLAGS += -DSCALARFLOAT
float: build

debug: CXXFLAGS += -g -DCUDATHREADSYNC
debug: NVCCFLAGS += -g -lineinfo -Xptxas --generate-line-info # -G
debug: build
build: $(CLASS_OBJS) $(CU_OBJS) $(PROG_MAINS)  $(PROGS)

clean: 
	rm -f $(PROG_OBJS) $(CLASS_OBJS) $(CU_OBJS) $(PROG_OBJS) $(PROG_MAINS)

print-%  : ; @echo $* = $($*)
