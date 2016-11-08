using namespace std;

//definitions needed for DelaunayLoc, voroguppy namespace, Triangle, Triangle, and all GPU functions, respectively

#define EPSILON 1e-12
#define dbl float
#define REAL double
#define ANSI_DECLARATIONS
#define ENABLE_CUDA

#include "spv2d.h"

SPV2D::SPV2D(int n)
    {
    printf("Initializing %i cells with random positions in a square box\n",n);
    initialize(n);
    };
