cmake_minimum_required(VERSION 3.9.0)

project(cellGPU LANGUAGES CUDA CXX)

add_definitions(-DENABLE_CUDA)

find_package(OpenMP REQUIRED)

set(CUDA_ARCH "30;35;50")
                #if you have different cuda-capable hardware, modify this line to get much more optimized performance. By default,
                #I have set this to work on Tesla K40s (still used at many XSEDE facilities), but add the correct codes for optimizing performance
                #on your cards


set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS}")
#note: CGAL (which I will get rid of someday) needs at least c++14; some advanced gcc compilers will throw errors if you don't set the cpp standard to 17, though
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -DBOOST_MATH_DISABLE_FLOAT128 -fopenmp -frounding-math")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -arch=sm_${CUDA_ARCH} -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakeHelp)
find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
            ${EIGEN3_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/inc
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/utility
            ${CMAKE_SOURCE_DIR}/src/updaters
            ${CMAKE_SOURCE_DIR}/src/simulation
            ${CMAKE_SOURCE_DIR}/src/models
            ${CMAKE_SOURCE_DIR}/src/analysis
            ${CMAKE_SOURCE_DIR}/src/databases
   )

add_subdirectory(src)

set(myLibs
    model
    modelGPU
    updaters
    updatersGPU
    analysis
    database
    simulation
    utility
    utilityGPU
    )

link_directories(${CUDA_LIBRARY_DIRS} $ENV{LD_LIBRARY_PATH})

set(SOURCES
    )
set(HEADERS
    )

# list the names of cpp files corresponding to linked executables you'd like...
foreach(ARG
        voronoi
        Vertex
        )
add_executable("${ARG}.out" "${ARG}.cpp" )
target_link_libraries("${ARG}.out"
    ${CUDA_LIBRARIES}
    ${myLibs}
    CGAL::CGAL
    netcdf
    netcdf_c++
    OpenMP::OpenMP_CXX
    )
endforeach()
