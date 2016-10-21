#ifndef DELCHECKGPU
#define DELCHECKGPU

using namespace std;
using namespace voroguppy;

#include "gpubox.h"
#include "gpuarray.h"
#include "gpucell.h"

class DelaunayTest
    {
    public:
        DelaunayTest(){};

        void testTriangulation(vector<float> &points,
                GPUArray<int> &circumcenters,
                dbl cellsize,
                gpubox &bx,
                GPUArray<bool> &reTriangulate
                );

    private:
        int N;

    };



#endif
