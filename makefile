#standard places to find cuda files
CUDA_INC = /usr/local/cuda/includes
CUDA_LIB = /usr/local/cuda/lib64
CUDA_LIB2 = /usr/local/cuda/lib

CXX := g++
CC := gcc
LINK := g++ #-fPIC
NVCC := nvcc

INCLUDES = -I. -I./src/ -I./ext_src/ -I./inc/ -I$(CUDA_INC)
LIB_CUDA = -L. -L$(CUDA_LIB) -L$(CUDA_LIB2) -lcuda -lcudart

#common flags
COMMONFLAGS += $(INCLUDES) -O3 -std=c++11 #-g
NVCCFLAGS += -D_FORCE_INLINES $(COMMONFLAGS) -lineinfo #-Xptxas -dlcm=ca#-G
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -w
CFLAGS += $(COMMONFLAGS)

#target rules
all:build

build: delGPU.out

OBJS= obj/voroguppy.o obj/DelaunayLoc.o obj/Delaunay1.o obj/DelaunayTri.o obj/gpucell.o obj/DelaunayCheckGPU.o obj/DelaunayMD.o obj/spv2d.o

EXT_OBJS = obj/triangle.o

CUOBJS= obj/gpucell.cu.o obj/DelaunayCheckGPU.cu.o obj/DelaunayMD.cu.o

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

obj/DelaunayMD.o:src/DelaunayMD.cpp obj/DelaunayMD.cu.o $(EXT_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/spv2d.o:src/spv2d.cpp obj/DelaunayMD.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayTri.o:src/DelaunayTri.cpp obj/triangle.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<


obj/Delaunay1.o:src/Delaunay1.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

obj/DelaunayLoc.o:src/DelaunayLoc.cpp obj/Delaunay1.o $(EXT_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

obj/voroguppy.o:voroguppy.cpp
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

delGPU.out: $(OBJS) $(CUOBJS) $(EXT_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ $+

run: build
	./delGPU.out

clean:
	rm -f $(OBJS) $(CUOBJS) delGPU.out

