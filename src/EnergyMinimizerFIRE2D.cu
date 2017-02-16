#define NVCC
#define ENABLE_CUDA

#include "EnergyMinimizerFIRE2D.cuh"

/*! \file EnergyMinimizerFIRE2D.cu
  defines kernel callers and kernels for GPU calculations related to FIRE minimization

 \addtogroup EnergyMinimizerFIRE2DKernels
 @{
 */

/*!
  set the first N elements of the d_velocity vector to 0.0
*/
__global__ void gpu_zero_velocity_kernel(Dscalar2 *d_velocity,
                                              int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    d_velocity[idx].x = 0.0;
    d_velocity[idx].y = 0.0;
    return;
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
update the velocity in a velocity Verlet step
*/
__global__ void gpu_update_velocity_kernel(Dscalar2 *d_velocity, Dscalar2 *d_force, Dscalar deltaT, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_velocity[idx].x += 0.5*deltaT*d_force[idx].x;
    d_velocity[idx].y += 0.5*deltaT*d_force[idx].y;
    };

/*!
update the velocity according to a FIRE step
*/
__global__ void gpu_update_velocity_FIRE_kernel(Dscalar2 *d_velocity, Dscalar2 *d_force, Dscalar alpha, Dscalar scaling, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_velocity[idx].x = (1-alpha)*d_velocity[idx].x + alpha*scaling*d_force[idx].x;
    d_velocity[idx].y = (1-alpha)*d_velocity[idx].y + alpha*scaling*d_force[idx].y;
    };

/*!
calculate the displacement in a velocity verlet step according to the force and velocity 
*/
__global__ void gpu_displacement_vv_kernel(Dscalar2 *d_displacement, Dscalar2 *d_velocity,
                                           Dscalar2 *d_force, Dscalar deltaT, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_displacement[idx].x = deltaT*d_velocity[idx].x+0.5*deltaT*deltaT*d_force[idx].x;
    d_displacement[idx].y = deltaT*d_velocity[idx].y+0.5*deltaT*deltaT*d_force[idx].y;
    };

/*!
  \param d_velocity the GPU array data of the velocities
  \param N length of the array
  \post all elements of d_velocity are set to (0.0,0.0)
  */
bool gpu_zero_velocity(Dscalar2 *d_velocity,
                    int N
                    )
    {
    //optimize block size later
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;
    gpu_zero_velocity_kernel<<<nblocks, block_size>>>(d_velocity,
                                                    N
                                                    );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

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
\param d_velocity Dscalar2 array of velocity
\param d_force Dscalar2 array of force
\param alpha the FIRE parameter
\param scaling the square root of (v.v / f.f)
\param N      the length of the arrays
\post v = (1-alpha)v + alpha*scalaing*force
*/
bool gpu_update_velocity_FIRE(Dscalar2 *d_velocity, Dscalar2 *d_force, Dscalar alpha, Dscalar scaling, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_update_velocity_FIRE_kernel<<<nblocks,block_size>>>(
                                                d_velocity,
                                                d_force,
                                                alpha,
                                                scaling,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
\param d_velocity Dscalar2 array of velocity
\param d_force Dscalar2 array of force
\param deltaT time step
\param N      the length of the arrays
\post v = v + 0.5*deltaT*force
*/
bool gpu_update_velocity(Dscalar2 *d_velocity, Dscalar2 *d_force, Dscalar deltaT, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_update_velocity_kernel<<<nblocks,block_size>>>(
                                                d_velocity,
                                                d_force,
                                                deltaT,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
\param d_displacement Dscalar2 array of displacements
\param d_velocity Dscalar2 array of velocities
\param d_force Dscalar2 array of forces
\param Dscalar deltaT the current time step
\param N      the length of the arrays
\post displacement = dt*velocity + 0.5 *dt^2*force
*/
bool gpu_displacement_velocity_verlet(Dscalar2 *d_displacement,
                      Dscalar2 *d_velocity,
                      Dscalar2 *d_force,
                      Dscalar deltaT,
                      int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_displacement_vv_kernel<<<nblocks,block_size>>>(
                                                d_displacement,d_velocity,d_force,deltaT,N);
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
a two-step parallel reduction alorithm that first does a partial sum reduction of input into the
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


/** @} */ //end of group declaration
