#ifndef LOCK_H
#define LOCK_H
struct Lock
    {
        int *mutex;
        Lock()
            {
            int state = 0;

            cudaMalloc((void**) &mutex,sizeof(int));
            cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
            };

        ~Lock()
            {
            cudaFree(mutex);
            };

        __device__ void lock()
            {
            while(atomicCAS(mutex,0,1) != 0);
            };
        __device__ void unlock()
            {
            atomicExch(mutex,0);
            };
    };


#endif

