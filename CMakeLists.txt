cmake_minimum_required(VERSION 3.22)

project(cellGPU LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
#note: CGAL (which I will get rid of someday) needs at least c++14; some advanced gcc compilers will throw errors if you don't set the cpp standard to 17, though
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
include(CMakePrintHelpers)

#set a default "Release" build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
endif()


find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED)
find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)



include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA found, building with GPU support")
    add_definitions(-DENABLE_CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    #if you have different cuda-capable hardware, modify this line to get much more optimized performance. By default,
    set(CMAKE_CUDA_ARCHITECTURES "50")
    message("cuda directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
else()
    message(STATUS "CUDA not found, building without GPU support")
endif()




# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DBOOST_MATH_DISABLE_FLOAT128 -fopenmp -frounding-math")

endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakeHelp)

include_directories(
            ${EIGEN3_INCLUDE_DIR}
            inc
            src/utility src/updaters src/simulation
            src/models src/analysis src/databases
   )

add_subdirectory(src)

set(myLibs
    model
    updaters
    analysis
    databases
    simulation
    utility
    )
list(APPEND myLibs 
    modelGPU
    updatersGPU
    utilityGPU
)
# link_directories(${CUDA_LIBRARY_DIRS} $ENV{LD_LIBRARY_PATH})

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
    ${myLibs}
    CGAL::CGAL
    OpenMP::OpenMP_CXX
    )
endforeach()
