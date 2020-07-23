#include "utilities.cuh"

/*! \file utilities.cu
  defines kernel callers and kernels for some simple GPU array calculations

 \addtogroup utilityKernels
 @{
 */

template <typename T>
__global__ void gpu_add_gpuarray_kernel(T *a, T *b, int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    a[idx] = a[idx]+b[idx];
    return;
    };


template<typename T>
bool gpu_add_gpuarray(GPUArray<T> &answer, GPUArray<T> &adder, int N, int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = (N)/block_size + 1;
    ArrayHandle<T> a(answer,access_location::device,access_mode::readwrite);
    ArrayHandle<T> b(adder,access_location::device,access_mode::read);
    gpu_add_gpuarray_kernel<<<nblocks,block_size>>>(a.data,b.data,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/*!
take two vectors and return a vector of double2s, where each entry is vec1[i].vec2[i]
*/
__global__ void gpu_dot_double_double2_vectors_kernel(double *d_vec1, double2 *d_vec2, double2 *d_ans, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_ans[idx] = d_vec1[idx]*d_vec2[idx];
    };

/*!
take two vectors of double2 and return a vector of doubles, where each entry is vec1[i].vec2[i]
*/
__global__ void gpu_dot_double2_vectors_kernel(double2 *d_vec1, double2 *d_vec2, double *d_ans, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_ans[idx] = d_vec1[idx].x*d_vec2[idx].x + d_vec1[idx].y*d_vec2[idx].y;
    };

/*!
\param d_vec1 double input array
\param d_vec2 double2 input array
\param d_ans  double2 output array... d_ans[idx] = d_vec1[idx] * d_vec2[idx]
\param N      the length of the arrays
\post d_ans = d_vec1.d_vec2
*/
bool gpu_dot_double_double2_vectors(double *d_vec1, double2 *d_vec2, double2 *d_ans, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_dot_double_double2_vectors_kernel<<<nblocks,block_size>>>(
                                                d_vec1,
                                                d_vec2,
                                                d_ans,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
\param d_vec1 double2 input array
\param d_vec2 double2 input array
\param d_ans  double output array... d_ans[idx] = d_vec1[idx].d_vec2[idx]
\param N      the length of the arrays
\post d_ans = d_vec1.d_vec2
*/
bool gpu_dot_double2_vectors(double2 *d_vec1, double2 *d_vec2, double *d_ans, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_dot_double2_vectors_kernel<<<nblocks,block_size>>>(
                                                d_vec1,
                                                d_vec2,
                                                d_ans,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
add the first N elements of array and put it in output[helperIdx]
*/
__global__ void gpu_serial_reduction_kernel(double *array, double *output, int helperIdx,int N)
    {
    double ans = 0.0;
    for (int i = 0; i < N; ++i)
        ans += array[i];
    output[helperIdx] = ans;
    return;
    };

/*!
add the first N elements of double2 array and put it in output[helperIdx]
*/
__global__ void gpu_serial_reduction_kernel(double2 *array, double2 *output, int helperIdx,int N)
    {
    double2 ans = make_double2(0.0,0.0);
    for (int i = 0; i < N; ++i)
        ans = ans + array[i];
    output[helperIdx] = ans;
    return;
    };

/*!
perform a block reduction, storing the partial sums of input into output
*/
__global__ void gpu_parallel_block_reduction_kernel(double *input, double *output,int N)
    {
    extern __shared__ double sharedArray[];

    unsigned int tidx = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    //load into shared memory and synchronize
    if(i < N)
        sharedArray[tidx] = input[i];
    else
        sharedArray[tidx] = 0.0;
    __syncthreads();

    //reduce
    for (int s = blockDim.x/2; s>0; s>>=1)
        {
        if (tidx < s)
            sharedArray[tidx] += sharedArray[tidx+s];
        __syncthreads();
        };
    //write to the correct block of the output array
    if (tidx==0)
        output[blockIdx.x] = sharedArray[0];
    };

/*!
a slight optimization of the previous block reduction, c.f. M. Harris presentation
*/
__global__ void gpu_parallel_block_reduction2_kernel(double *input, double *output,int N)
    {
    extern __shared__ double sharedArray[];

    unsigned int tidx = threadIdx.x;
    unsigned int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    double sum;
    //load into shared memory and synchronize
    if(i < N)
        sum = input[i];
    else
        sum = 0.0;
    if(i + blockDim.x < N)
        sum += input[i+blockDim.x];

    sharedArray[tidx] = sum;
    __syncthreads();

    //reduce
    for (int s = blockDim.x/2; s>0; s>>=1)
        {
        if (tidx < s)
            sharedArray[tidx] = sum = sum+sharedArray[tidx+s];
        __syncthreads();
        };
    //write to the correct block of the output array
    if (tidx==0)
        output[blockIdx.x] = sum;
    };

/*!
block reduction for double2 arrays, c.f. M. Harris presentation
*/
__global__ void gpu_parallel_block_reduction2_kernel(double2 *input, double2 *output,int N)
    {
    extern __shared__ double2 sharedArray2[];

    unsigned int tidx = threadIdx.x;
    unsigned int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    double2 sum;
    //load into shared memory and synchronize
    if(i < N)
        sum = input[i];
    else
        sum = make_double2(0.0,0.0);
    if(i + blockDim.x < N)
        sum = sum + input[i+blockDim.x];

    sharedArray2[tidx] = sum;
    __syncthreads();

    //reduce
    for (int s = blockDim.x/2; s>0; s>>=1)
        {
        if (tidx < s)
            sharedArray2[tidx] = sum = sum+sharedArray2[tidx+s];
        __syncthreads();
        };
    //write to the correct block of the output array
    if (tidx==0)
        output[blockIdx.x] = sum;
    };

/*!
a two-step parallel reduction algorithm that first does a partial sum reduction of input into the
intermediate array, then launches a second kernel to sum reduce intermediate into output[helperIdx]
\param input the input array to sum
\param intermediate an array that input is block-reduced to
\param output the intermediate array will be sum reduced and stored in one of the components of output
\param helperIdx the location in output to store the answer
\param N the size of the input and  intermediate arrays
*/
bool gpu_parallel_reduction(double *input, double *intermediate, double *output, int helperIdx, int N)
    {
    unsigned int block_size = 256;
    unsigned int nblocks  = N/block_size + 1;
    //first do a block reduction of input
    unsigned int smem = block_size*sizeof(double);

    //Do a block reduction of the input array
    gpu_parallel_block_reduction2_kernel<<<nblocks,block_size,smem>>>(input,intermediate, N);
    HANDLE_ERROR(cudaGetLastError());

    //sum reduce the temporary array, saving the result in the right slot of the output array
    gpu_serial_reduction_kernel<<<1,1>>>(intermediate,output,helperIdx,nblocks);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
a two-step parallel reduction algorithm for double2's that first does a partial sum reduction of input into the
intermediate array, then launches a second kernel to sum reduce intermediate into output[helperIdx]
\param input the input array to sum
\param intermediate an array that input is block-reduced to
\param output the intermediate array will be sum reduced and stored in one of the components of output
\param helperIdx the location in output to store the answer
\param N the size of the input and  intermediate arrays
*/
bool gpu_parallel_reduction(double2 *input, double2 *intermediate, double2 *output, int helperIdx, int N)
    {
    unsigned int block_size = 256;
    unsigned int nblocks  = N/block_size + 1;
    //first do a block reduction of input
    unsigned int smem = block_size*sizeof(double);

    //Do a block reduction of the input array
    gpu_parallel_block_reduction2_kernel<<<nblocks,block_size,smem>>>(input,intermediate, N);
    HANDLE_ERROR(cudaGetLastError());

    //sum reduce the temporary array, saving the result in the right slot of the output array
    gpu_serial_reduction_kernel<<<1,1>>>(intermediate,output,helperIdx,nblocks);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
This serial reduction routine should probably never be called. It provides an interface to the
gpu_serial_reduction_kernel above that may be useful for testing
  */
bool gpu_serial_reduction(double *array, double *output, int helperIdx, int N)
    {
    gpu_serial_reduction_kernel<<<1,1>>>(array,output,helperIdx,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

template <typename T>
__global__ void gpu_set_array_kernel(T *arr,T value, int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    arr[idx] = value;
    return;
    };

template<typename T>
bool gpu_set_array(T *array, T value, int N,int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;
    gpu_set_array_kernel<<<nblocks, block_size>>>(array,value,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

template <typename T>
__global__ void gpu_copy_gpuarray_kernel(T *copyInto,T *copyFrom, int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    copyInto[idx] = copyFrom[idx];
    return;
    };

template<typename T>
bool gpu_copy_gpuarray(GPUArray<T> &copyInto,GPUArray<T> &copyFrom,int numberOfElementsToCopy,int maxBlockSize)
    {
    int N = copyFrom.getNumElements();
    if(numberOfElementsToCopy >0)
        N = numberOfElementsToCopy;
    if(copyInto.getNumElements() < N)
        copyInto.resize(N);
    unsigned int block_size = maxBlockSize;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = (N)/block_size + 1;
    ArrayHandle<T> ci(copyInto,access_location::device,access_mode::overwrite);
    ArrayHandle<T> cf(copyFrom,access_location::device,access_mode::read);
    gpu_copy_gpuarray_kernel<<<nblocks,block_size>>>(ci.data,cf.data,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

//Declare templates used...cuda is annoying sometimes
template bool gpu_copy_gpuarray<double>(GPUArray<double> &copyInto,GPUArray<double> &copyFrom,int n, int maxBlockSize);
template bool gpu_copy_gpuarray<double2>(GPUArray<double2> &copyInto,GPUArray<double2> &copyFrom,int n, int maxBlockSize);
template bool gpu_copy_gpuarray<int>(GPUArray<int> &copyInto,GPUArray<int> &copyFrom,int n, int maxBlockSize);
template bool gpu_copy_gpuarray<int3>(GPUArray<int3> &copyInto,GPUArray<int3> &copyFrom,int n, int maxBlockSize);

template bool gpu_set_array<int>(int *,int, int, int);
template bool gpu_set_array<unsigned int>(unsigned int *,unsigned int, int, int);
template bool gpu_set_array<int2>(int2 *,int2, int, int);
template bool gpu_set_array<int3>(int3 *,int3, int, int);
template bool gpu_set_array<double>(double *,double, int, int);
template bool gpu_set_array<double2>(double2 *,double2, int, int);

template bool gpu_add_gpuarray<double>(GPUArray<double> &answer, GPUArray<double> &adder, int N, int maxBlockSize);
template bool gpu_add_gpuarray<double2>(GPUArray<double2> &answer, GPUArray<double2> &adder, int N, int maxBlockSize);
/** @} */ //end of group declaration
