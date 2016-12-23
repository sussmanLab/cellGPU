#ifndef INDEXER
#define INDEXER


#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*!
 * A class for converting between a 2d index and a 1-d array, which makes calculation on
 * the GPU a bit easier. This was inspired by the indexer class of Hoomd-blue
 */
class Index2D
    {
    public:
        HOSTDEVICE Index2D(unsigned int w=0) : width(w), height(w) {}
        HOSTDEVICE Index2D(unsigned int w, unsigned int h) : width(w), height(h) {}

        HOSTDEVICE unsigned int operator()(unsigned int i, unsigned int j) const
            {
            return j*width + i;
            }
        //!Return the number of elements that the indexer can index
        HOSTDEVICE unsigned int getNumElements() const
            {
            return width*height;
            }

        //!Get the width
        HOSTDEVICE unsigned int getW() const
            {
            return width;
            }

        //!get the height
        HOSTDEVICE unsigned int getH() const
            {
            return height;
            }

    private:
        unsigned int width;   //!< array width
        unsigned int height;   //!< array height
    };

#undef HOSTDEVICE
#endif
