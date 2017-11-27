#define NVCC
#define ENABLE_CUDA

#include "utilities.cuh"

/*! \file utilities.cu
  defines kernel callers and kernels for some simple GPU array calculations

 \addtogroup utilityKernels
 @{
 */

/*!
take two vectors and return a vector of Dscalar2s, where each entry is vec1[i].vec2[i]
*/
__global__ void gpu_dot_Dscalar_Dscalar2_vectors_kernel(Dscalar *d_vec1, Dscalar2 *d_vec2, Dscalar2 *d_ans, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_ans[idx] = d_vec1[idx]*d_vec2[idx];
    };

/*!
take two vectors of Dscalar2 and return a vector of Dscalars, where each entry is vec1[i].vec2[i]
*/
__global__ void gpu_dot_Dscalar2_vectors_kernel(Dscalar2 *d_vec1, Dscalar2 *d_vec2, Dscalar *d_ans, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_ans[idx] = d_vec1[idx].x*d_vec2[idx].x + d_vec1[idx].y*d_vec2[idx].y;
    };

/*!
\param d_vec1 Dscalar input array
\param d_vec2 Dscalar2 input array
\param d_ans  Dscalar2 output array... d_ans[idx] = d_vec1[idx] * d_vec2[idx]
\param N      the length of the arrays
\post d_ans = d_vec1.d_vec2
*/
bool gpu_dot_Dscalar_Dscalar2_vectors(Dscalar *d_vec1, Dscalar2 *d_vec2, Dscalar2 *d_ans, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_dot_Dscalar_Dscalar2_vectors_kernel<<<nblocks,block_size>>>(
                                                d_vec1,
                                                d_vec2,
                                                d_ans,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
\param d_vec1 Dscalar2 input array
\param d_vec2 Dscalar2 input array
\param d_ans  Dscalar output array... d_ans[idx] = d_vec1[idx].d_vec2[idx]
\param N      the length of the arrays
\post d_ans = d_vec1.d_vec2
*/
bool gpu_dot_Dscalar2_vectors(Dscalar2 *d_vec1, Dscalar2 *d_vec2, Dscalar *d_ans, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_dot_Dscalar2_vectors_kernel<<<nblocks,block_size>>>(
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
__global__ void gpu_serial_reduction_kernel(Dscalar *array, Dscalar *output, int helperIdx,int N)
    {
    Dscalar ans = 0.0;
    for (int i = 0; i < N; ++i)
        ans += array[i];
    output[helperIdx] = ans;
    return;
    };

/*!
add the first N elements of Dscalar2 array and put it in output[helperIdx]
*/
__global__ void gpu_serial_reduction_kernel(Dscalar2 *array, Dscalar2 *output, int helperIdx,int N)
    {
    Dscalar2 ans = make_Dscalar2(0.0,0.0);
    for (int i = 0; i < N; ++i)
        ans = ans + array[i];
    output[helperIdx] = ans;
    return;
    };

/*!
perform a block reduction, storing the partial sums of input into output
*/
__global__ void gpu_parallel_block_reduction_kernel(Dscalar *input, Dscalar *output,int N)
    {
    extern __shared__ Dscalar sharedArray[];

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
__global__ void gpu_parallel_block_reduction2_kernel(Dscalar *input, Dscalar *output,int N)
    {
    extern __shared__ Dscalar sharedArray[];

    unsigned int tidx = threadIdx.x;
    unsigned int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    Dscalar sum;
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
block reduction for Dscalar2 arrays, c.f. M. Harris presentation
*/
__global__ void gpu_parallel_block_reduction2_kernel(Dscalar2 *input, Dscalar2 *output,int N)
    {
    extern __shared__ Dscalar2 sharedArray2[];

    unsigned int tidx = threadIdx.x;
    unsigned int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    Dscalar2 sum;
    //load into shared memory and synchronize
    if(i < N)
        sum = input[i];
    else
        sum = make_Dscalar2(0.0,0.0);
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
bool gpu_parallel_reduction(Dscalar *input, Dscalar *intermediate, Dscalar *output, int helperIdx, int N)
    {
    unsigned int block_size = 256;
    unsigned int nblocks  = N/block_size + 1;
    //first do a block reduction of input
    unsigned int smem = block_size*sizeof(Dscalar);

    //Do a block reduction of the input array
    gpu_parallel_block_reduction2_kernel<<<nblocks,block_size,smem>>>(input,intermediate, N);
    HANDLE_ERROR(cudaGetLastError());

    //sum reduce the temporary array, saving the result in the right slot of the output array
    gpu_serial_reduction_kernel<<<1,1>>>(intermediate,output,helperIdx,nblocks);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
a two-step parallel reduction algorithm for Dscalar2's that first does a partial sum reduction of input into the
intermediate array, then launches a second kernel to sum reduce intermediate into output[helperIdx]
\param input the input array to sum
\param intermediate an array that input is block-reduced to
\param output the intermediate array will be sum reduced and stored in one of the components of output
\param helperIdx the location in output to store the answer
\param N the size of the input and  intermediate arrays
*/
bool gpu_parallel_reduction(Dscalar2 *input, Dscalar2 *intermediate, Dscalar2 *output, int helperIdx, int N)
    {
    unsigned int block_size = 256;
    unsigned int nblocks  = N/block_size + 1;
    //first do a block reduction of input
    unsigned int smem = block_size*sizeof(Dscalar);

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
bool gpu_serial_reduction(Dscalar *array, Dscalar *output, int helperIdx, int N)
    {
    gpu_serial_reduction_kernel<<<1,1>>>(array,output,helperIdx,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
