#ifndef INDEXER
#define INDEXER


#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

//a class for converting between a 2d index and a 1-d array

class Index2D
    {
    public:
        HOSTDEVICE Index2D(unsigned int w=0) : width(w), height(w) {}
        HOSTDEVICE Index2D(unsigned int w, unsigned int h) : width(w), height(h) {}

        HOSTDEVICE unsigned int operator()(unsigned int i, unsigned int j) const
            {
            return j*width + i;
            }

        HOSTDEVICE unsigned int getNumElements() const
            {
            return width*height;
            }

        HOSTDEVICE unsigned int getW() const
            {
            return width;
            }

        HOSTDEVICE unsigned int getH() const
            {
            return height;
            }

    private:
        unsigned int width;   // array width
        unsigned int height;   // array height
    };

#undef HOSTDEVICE
#endif


