#ifndef INDEXER
#define INDEXER
/*
This file is based on part of the HOOMD-blue project, released under the BSD 3-Clause License:

HOOMD-blue Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer both in the code and prominently in any materials provided with the distribution.
3. Neither the name ofthe copyright holder nor the names of its contributors may be used to enorse or promote products derived from this software without specific prior written permission

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//As you might suspect from the above, the classes and structures in this file are modifications of the Index1D.h file from the HOOMD-Blue package.
//Credit to Joshua A. Anderson

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file indexer.h */
//!Switch between a 2D array to a flattened, 1D index
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

        unsigned int width;   //!< array width
        unsigned int height;   //!< array height
    };

#undef HOSTDEVICE
#endif
