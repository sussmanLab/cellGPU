#standard places to find cuda files
CUDA_INC = /usr/local/cuda/includes
CUDA_LIB = /usr/local/cuda/lib64
CUDA_LIB2 = /usr/local/cuda/lib

CXX := g++
CC := gcc
LINK := g++ #-fPIC
NVCC := nvcc

INCLUDES = -I. -I./src/ -I./ext_src/ -I./inc/ -I$(CUDA_INC) -I/home/user/CGAL/CGAL-4.9/include
LIB_CUDA = -L. -L$(CUDA_LIB) -L$(CUDA_LIB2) -lcuda -lcudart
LIB_CGAL = -L/home/user/CGAL/CGAL-4.9/lib -lCGAL -lCGAL_Core -lgmp -lmpfr
LIB_NETCDF = -lnetcdf_c++ -lnetcdf

#common flags
COMMONFLAGS += $(INCLUDES) -O3 -std=c++11 -g
NVCCFLAGS += -D_FORCE_INLINES $(COMMONFLAGS) -lineinfo -Wno-deprecated-gpu-targets #-Xptxas -dlcm=ca#-G
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -w -frounding-math
CFLAGS += $(COMMONFLAGS) -frounding-math

#target rules
all:build

build: delGPU.out

OBJS= obj/voroguppy.o obj/DelaunayLoc.o obj/Delaunay1.o obj/DelaunayTri.o obj/DelaunayCGAL.o obj/gpucell.o obj/DelaunayCheckGPU.o obj/DelaunayMD.o obj/spv2d.o

EXT_OBJS = obj/triangle.o

CUOBJS= obj/gpucell.cu.o obj/DelaunayCheckGPU.cu.o obj/DelaunayMD.cu.o obj/spv2d.cu.o

#for now, just compile triangle separately and copy the .o file to /obj directory
#TRILIBDEFS = -DTRILIBRARY
#CSWITCHES = -O
#obj/triangle.o: ext_src/triangle.c ext_src/triangle.h
#	$(CC) $(CSWITCHES) $(INCLUDES) -o $@ -c $<


obj/gpucell.cu.o:src/gpucell.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA)  -o $@ -c $<

obj/gpucell.o:src/gpucell.cpp obj/gpucell.cu.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayCheckGPU.cu.o:src/DelaunayCheckGPU.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA)  -o $@ -c $<

obj/DelaunayCheckGPU.o:src/DelaunayCheckGPU.cpp obj/DelaunayCheckGPU.cu.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayMD.cu.o:src/DelaunayMD.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayMD.o:src/DelaunayMD.cpp obj/DelaunayMD.cu.o obj/DelaunayCGAL.o $(EXT_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<

obj/spv2d.cu.o:src/spv2d.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/spv2d.o:src/spv2d.cpp obj/DelaunayMD.o obj/spv2d.cu.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayTri.o:src/DelaunayTri.cpp obj/triangle.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

obj/DelaunayCGAL.o:src/DelaunayCGAL.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_CGAL) -o $@ -c $<

obj/Delaunay1.o:src/Delaunay1.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

obj/DelaunayLoc.o:src/DelaunayLoc.cpp obj/Delaunay1.o obj/DelaunayCGAL.o $(EXT_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_CGAL) -o $@ -c $<

obj/voroguppy.o:voroguppy.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<

delGPU.out: $(OBJS) $(CUOBJS) $(EXT_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+

run: build
	./delGPU.out

clean:
	rm -f $(OBJS) $(CUOBJS) delGPU.out

