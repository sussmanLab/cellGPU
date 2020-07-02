#ifndef GPUARRAY_H
#define GPUARRAY_H

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

//As you might suspect from the above, the classes and structures in this file are modifications of the GPUArray.h file from the HOOMD-Blue package.
//Credit to Joshua A. Anderson

/*!\file gpuarray.h */
/*!
This file defines two helpful classes for working with data on both th CPU and GPU.
GPUArray<T> is a templated array that carries around with it some data, as well as
information about where that data was last modified and/or accessed. It can be
dynamically resized, but does not have vector methods like push_back.

GPUArray<T> objects are manipulated by ArrayHandle<T> objects. So, if you have declared a
GPUArray<int> cellIndex(numberOfCells)
somewhere, you can access that data by on the spot creating an ArrayHandle:
ArrayHandle<int> h_ci(cellPositions,access_location::host, access_mode::overwrite);
The data can then be accessed like
for (int c = 0; c < numberOfCells;++c)
    h_ci.data[c] = .....
*/
// for vector types
#include "std_include.h"
#include <cuda_runtime.h>


//!A structure for declaring where we want to access data
struct access_location
    {
    //!An enumeration of possibilities
    enum Enum
        {
        host,   //!<We want to access the data on the CPU
        device  //!<We want to access the data on the GPU
        };
    };

//!A structure for declaring where the current version of the data is
struct data_location
    {
    //!An enumeration of possibilities
    enum Enum
        {
        host,       //!< data was last modified on host
        device,     //!< data was last modified on device
        hostdevice  //!< data is current on both host and device
        };
    };

//!A structure for declaring how we want to access data (read, write, overwrite?)
struct access_mode
    {
    //!An enumeration of possibilities
    enum Enum
        {
        read,       //!< we just want to read
        readwrite,  //!< we intend to both read and write
        overwrite   //!< we will completely overwrite all of the data
        };
    };

//!A class for handling data simultaneously on the CPU and GPU
/*!This class and accessor are based on GPUArray.h, from the HOOMD-Blue simulation package.
It is, however, simplified. It takes care of cuda memory copying for templated arrays.
A flag (default to false) when declaring a GPUArray controls whether the memory is HostRegistered
but only handles synchronous copy operatations (no Asynch, no HostRegister, etc.)
It is also only for 1D arrays of data. Importantly, the user accesses and handles data through the ArrayHandle class.
*/
template<class T> class GPUArray;

//!A mechanism for accessing data in GPUArray objects's
/*!
The ArrayHandle, well, handles the data in the GPUArray. Given, e.g., an
ArrayHandle<int> h(gpuarray,access_location::host, access_mode::readwrite);
The user would access one of the integers via h.data[i] on the host or a direct acces on the device
*/
template<class T> class ArrayHandle
    {
    public:
        //!the only constructor takes a reference to the GPUArray, a location and a mode
        inline ArrayHandle(const GPUArray<T>& gpu_array, const access_location::Enum location = access_location::host,
                           const access_mode::Enum mode = access_mode::readwrite);
        inline ~ArrayHandle();

        T* data;          //!< a pointer to the GPUArray's data

        void operator=(const ArrayHandle& rhs)
                {
                data=rhs.data;
                };

    private:
        const GPUArray<T>& gpu_array; //!< The GPUarray that the Handle was initialized with
    };

//GPUArray, a class for managing a 1d array of elements on the GPU and the CPU simultaneously. The array has a flat data pointer with some number of elements, keeping a copy on both the host and device. An ArrayHandle instance allows access to the data, which either simply returns the pointer (if the data was last changed from the same location) or first copied over and then returned.
template<class T> class GPUArray
    {
    public:
        GPUArray(bool _register=false);
        //! The most common constructor takes in the desired size of the array
        GPUArray(unsigned int num_elements,bool _register=false);
        virtual ~GPUArray();

        GPUArray(const GPUArray& from);
        GPUArray& operator=(const GPUArray& rhs);
        //!Swap two GPUarrays efficiently
        inline void swap(GPUArray& from);
        //!Get the size of the array
        unsigned int getNumElements() const
            {
            return Num_elements;
            }
        //! Switch from simple memcpys to HostRegister pinned memory copies. Not currently fully functional
        void setRegistered(bool _reg)
            {
            RegisterArray=_reg;
            if(RegisterArray)
                cudaHostRegister(h_data,Num_elements*sizeof(T),cudaHostRegisterDefault);
            };
        //!Resize the array...performs operations on both the CPU and GPU
        virtual void resize(unsigned int num_elements);

    protected:
        inline void memclear(unsigned int first=0);

        inline T* acquire(const access_location::Enum location, const access_mode::Enum mode) const;

        inline void release() const
            {
            Acquired = false;
            }

    private:
        mutable unsigned int Num_elements;            //!< Number of elements
        mutable bool Acquired;                //!< Tracks whether the data has been acquired
        bool RegisterArray;                //!< Tracks whether the data has been acquired
        mutable data_location::Enum Data_location;    //!< Tracks the current location of the data

    protected:
#ifdef ENABLE_CUDA
        mutable T* d_data; //!<pointer to memory on device
#endif
        mutable T* h_data; //!<pointer to memory on host

    private:
        inline void allocate();
        inline void deallocate();

#ifdef ENABLE_CUDA
        inline void memcpyDeviceToHost() const;
        inline void memcpyHostToDevice() const;
#endif

        inline T* resizeHostArray(unsigned int num_elements);

        inline T* resizeDeviceArray(unsigned int num_elements);

        //needs to be friends with ArrayHandle for this all to work
        friend class ArrayHandle<T>;
    };

// ******************************************
// ArrayHandle implementation
// *****************************************
template<class T> ArrayHandle<T>::ArrayHandle(const GPUArray<T>& _gpu_array, const access_location::Enum location,
                                              const access_mode::Enum mode) :
        data(_gpu_array.acquire(location, mode)), gpu_array(_gpu_array)
    {
    }

template<class T> ArrayHandle<T>::~ArrayHandle()
    {
    gpu_array.Acquired = false;
    }

// ******************************************
// GPUArray implementation
// *****************************************
template<class T> GPUArray<T>::GPUArray(bool _register) :
        Num_elements(0), Acquired(false), Data_location(data_location::host), RegisterArray(_register),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    }

template<class T> GPUArray<T>::GPUArray(unsigned int num_elements, bool _register) :
        Num_elements(num_elements), Acquired(false), Data_location(data_location::host), RegisterArray(_register),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    // allocate and clear memory
    allocate();
    memclear();
    }

template<class T> GPUArray<T>::~GPUArray()
    {
    deallocate();
    }

template<class T> GPUArray<T>::GPUArray(const GPUArray& from) : Num_elements(from.Num_elements), 
        Acquired(false), Data_location(data_location::host),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    // allocate and clear new memory the same size as the data in from
    allocate();
    memclear();

    // copy over the data to the new GPUArray
    if (Num_elements > 0)
        {
        ArrayHandle<T> h_handle(from, access_location::host, access_mode::read);
        memcpy(h_data, h_handle.data, sizeof(T)*Num_elements);
        }
    }

template<class T> GPUArray<T>& GPUArray<T>::operator=(const GPUArray& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        // free current memory
        deallocate();

        // is the array registered
        RegisterArray = rhs.RegisterArray;

        // copy over basic elements
        Num_elements = rhs.Num_elements;

        // initialize state variables
        Data_location = data_location::host;

        // allocate and clear new memory the same size as the data in rhs
        allocate();
        memclear();

        // copy over the data to the new GPUArray
        if (Num_elements > 0)
            {
            ArrayHandle<T> h_handle(rhs, access_location::host, access_mode::read);
            memcpy(h_data, h_handle.data, sizeof(T)*Num_elements);
            }
        }

    return *this;
    }

/*!
    a.swap(b) is:
        GPUArray c(a);
        a = b;
        b = c;
    It just swaps internal pointers
*/
template<class T> void GPUArray<T>::swap(GPUArray& from)
    {
    std::swap(Num_elements, from.Num_elements);
    std::swap(Acquired, from.Acquired);
    std::swap(Data_location, from.Data_location);
    std::swap(RegisterArray,from.RegisterArray);
#ifdef ENABLE_CUDA
    std::swap(d_data, from.d_data);
#endif
    std::swap(h_data, from.h_data);
    }

template<class T> void GPUArray<T>::allocate()
    {
    // don't allocate anything if there are zero elements
    if (Num_elements == 0)
        return;
    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    int retval = posix_memalign((void**)&h_data, 32, Num_elements*sizeof(T));
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating GPUArray.");
        }

#ifdef ENABLE_CUDA
//    if(RegisterArray)
//        cudaHostRegister(h_data,Num_elements*sizeof(T),cudaHostRegisterDefault);
    cudaMalloc(&d_data, Num_elements*sizeof(T));
#endif
    }

template<class T> void GPUArray<T>::deallocate()
    {
    // don't do anything if there are no elements
    if (Num_elements == 0)
        return;
    // free memory
#ifdef ENABLE_CUDA
    cudaFree(d_data);
//    if(RegisterArray)
//        cudaHostUnregister(h_data);
#endif

    free(h_data);

    // set pointers to NULL
    h_data = NULL;
#ifdef ENABLE_CUDA
    d_data = NULL;
#endif
    }

template<class T> void GPUArray<T>::memclear(unsigned int first)
    {
    // don't do anything if there are no elements
    if (Num_elements == 0)
        return;

    // clear memory
    memset(h_data+first, 0, sizeof(T)*(Num_elements-first));

#ifdef ENABLE_CUDA
    cudaMemset(d_data+first, 0, (Num_elements-first)*sizeof(T));
#endif
    }


#ifdef ENABLE_CUDA
template<class T> void GPUArray<T>::memcpyDeviceToHost() const
    {
    // don't do anything if there are no elements
    if (Num_elements == 0)
        return;


    cudaMemcpy(h_data, d_data, sizeof(T)*Num_elements, cudaMemcpyDeviceToHost);

    }

template<class T> void GPUArray<T>::memcpyHostToDevice() const
    {
    // don't do anything if there are no elements
    if (Num_elements == 0)
        return;

    cudaMemcpy(d_data, h_data, sizeof(T)*Num_elements, cudaMemcpyHostToDevice);
    }
#endif

/*!
    Acquire does all the work, keeping track of when data needs to be copied, etc.
    It is called by the ArrayHandle class
*/
template<class T> T* GPUArray<T>::acquire(const access_location::Enum location, const access_mode::Enum mode) const
    {
    Acquired = true;

    // (1) where do we want the data? (2) where *is* the data? (3) copy if necessary
    // if only reading, often avoid a copy
    if (location == access_location::host)
        {
        if (Data_location == data_location::host)
            {
            return h_data;
            }
#ifdef ENABLE_CUDA
        else if (Data_location == data_location::hostdevice)
            {
            if (mode == access_mode::read)
                Data_location = data_location::hostdevice;
            else if (mode == access_mode::readwrite)
                Data_location = data_location::host;
            else if (mode == access_mode::overwrite)
                Data_location = data_location::host;
            else
                {
                throw std::runtime_error("Error acquiring data7");
                }

            return h_data;
            }
        else if (Data_location == data_location::device)
            {
            if (mode == access_mode::read)
                {
                memcpyDeviceToHost();
                Data_location = data_location::hostdevice;
                }
            else if (mode == access_mode::readwrite)
                {
                memcpyDeviceToHost();
                Data_location = data_location::host;
                }
            else if (mode == access_mode::overwrite)
                {
                Data_location = data_location::host;
                }
            else
                {
                throw std::runtime_error("Error acquiring data6");
                }

            return h_data;
            }
#endif
        else
            {
            throw std::runtime_error("Error acquiring data5");
            }
        }
#ifdef ENABLE_CUDA
    else if (location == access_location::device)
        {
        if (Data_location == data_location::host)
            {
            if (mode == access_mode::read)
                {
                memcpyHostToDevice();
                Data_location = data_location::hostdevice;
                }
            else if (mode == access_mode::readwrite)
                {
                memcpyHostToDevice();
                Data_location = data_location::device;
                }
            else if (mode == access_mode::overwrite)
                {
                Data_location = data_location::device;
                }
            else
                {
                throw std::runtime_error("Error acquiring data4");
                }

            return d_data;
            }
        else if (Data_location == data_location::hostdevice)
            {
            if (mode == access_mode::read)
                Data_location = data_location::hostdevice;
            else if (mode == access_mode::readwrite)
                Data_location = data_location::device;
            else if (mode == access_mode::overwrite)
                Data_location = data_location::device;
            else
                {
                throw std::runtime_error("Error acquiring data3");
                }
            return d_data;
            }
        else if (Data_location == data_location::device)
            {
            return d_data;
            }
        else
            {
            throw std::runtime_error("Error acquiring data2");
            }
        }
#endif
    else
        {
        throw std::runtime_error("Error acquiring data1");
        }
    }

template<class T> T* GPUArray<T>::resizeHostArray(unsigned int num_elements)
    {
    // allocate resized array
    T *h_tmp = NULL;

    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    int retval = posix_memalign((void**)&h_tmp, 32, num_elements*sizeof(T));
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating GPUArray.");
        }

#ifdef ENABLE_CUDA
//    if(RegisterArray)
//        cudaHostRegister(h_tmp,Num_elements*sizeof(T),cudaHostRegisterDefault);
#endif

    // clear memory
    memset(h_tmp, 0, sizeof(T)*num_elements);

    // copy over data
    unsigned int num_copy_elements = Num_elements > num_elements ? num_elements : Num_elements;
    memcpy(h_tmp, h_data, sizeof(T)*num_copy_elements);

#ifdef ENABLE_CUDA
//    if(RegisterArray)
//        cudaHostUnregister(h_data);
#endif

    // free old memory location
    free(h_data);
    h_data = h_tmp;

    return h_data;
    }

template<class T> T* GPUArray<T>::resizeDeviceArray(unsigned int num_elements)
    {
#ifdef ENABLE_CUDA
    // allocate resized array
    T *d_tmp;
    cudaMalloc(&d_tmp, num_elements*sizeof(T));

    // clear memory
    cudaMemset(d_tmp, 0, num_elements*sizeof(T));

    // copy over data
    unsigned int num_copy_elements = Num_elements > num_elements ? num_elements : Num_elements;
    cudaMemcpy(d_tmp, d_data, sizeof(T)*num_copy_elements,cudaMemcpyDeviceToDevice);

    // free old memory location
    cudaFree(d_data);

    d_data = d_tmp;
    return d_data;
#else
    return NULL;
#endif
    }

template<class T> void GPUArray<T>::resize(unsigned int num_elements)
    {
    resizeHostArray(num_elements);
#ifdef ENABLE_CUDA
    resizeDeviceArray(num_elements);
#endif
    Num_elements = num_elements;
    }

#endif
