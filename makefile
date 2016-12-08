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
COMMONFLAGS += $(INCLUDES) -std=c++11 -g -DCGAL_DISABLE_ROUNDING_MATH_CHECK
NVCCFLAGS += -D_FORCE_INLINES $(COMMONFLAGS) -lineinfo -Wno-deprecated-gpu-targets -O3 -Xptxas --generate-line-info #-fmad=false#-O0#-dlcm=ca#-G
#COMMONFLAGS += $(INCLUDES) -std=c++11 -DCGAL_DISABLE_ROUNDING_MATH_CHECK
#NVCCFLAGS += -D_FORCE_INLINES $(COMMONFLAGS)  -Wno-deprecated-gpu-targets -O3 -Xptxas  #-fmad=false#-O0#-dlcm=ca#-G
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -w -frounding-math -O3
CFLAGS += $(COMMONFLAGS) -frounding-math

#target rules
all:build

build: delGPU.out

PROG_OBJS= obj/runellipse.o obj/voroguppy.o obj/runplates.o obj/runMakeDatabase.o

CLASS_OBJS= obj/DelaunayLoc.o obj/Delaunay1.o obj/DelaunayCGAL.o obj/gpucell.o obj/DelaunayMD.o obj/spv2d.o

CUOBJS= obj/gpucell.cu.o obj/DelaunayMD.cu.o obj/spv2d.cu.o


obj/gpucell.cu.o:src/gpucell.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA)  -o $@ -c $<

obj/gpucell.o:src/gpucell.cpp obj/gpucell.cu.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayMD.cu.o:src/DelaunayMD.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayMD.o:src/DelaunayMD.cpp obj/DelaunayMD.cu.o obj/DelaunayCGAL.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<

obj/spv2d.cu.o:src/spv2d.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/spv2d.o:src/spv2d.cpp obj/DelaunayMD.o obj/spv2d.cu.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayCGAL.o:src/DelaunayCGAL.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CGAL) $(LIB_CUDA) -o $@ -c $<

obj/Delaunay1.o:src/Delaunay1.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) -o $@ -c $<

obj/DelaunayLoc.o:src/DelaunayLoc.cpp obj/Delaunay1.o obj/DelaunayCGAL.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) -o $@ -c $<

###
#Programs
##
obj/voroguppy.o:voroguppy.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<

obj/runellipse.o:runellipse.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<

obj/runplates.o:runplates.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<

obj/runMakeDatabase.o:runMakeDatabase.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_NETCDF) -o $@ -c $<



delGPU.out: obj/voroguppy.o $(CLASS_OBJS) $(CUOBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_CUDA) $(LIB_CGAL) $(LIB_NETCDF) -o $@ $+

run: build
	./delGPU.out

clean:
	rm -f $(PROG_OBJS) $(CLASS_OBJS) $(CUOBJS) delGPU.out

