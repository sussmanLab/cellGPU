#include <cuda_runtime.h>
#include "cellListGPU.cuh"
#include "indexer.h"
#include "periodicBoundaries.h"
#include "functions.h"
#include <iostream>
#include <stdio.h>
#include "voronoiModelBase.cuh"
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

/*! \file voronoiModelBase.cu */
/*!
    \addtogroup voronoiModelBaseKernels
    @{
*/

/*!
  Independently check every triangle in the Delaunay mesh to see if the cirumcircle defined by the
  vertices of that triangle is empty. Use the cell list to ensure that only checks of nearby
  particles are required.
  */
__global__ void gpu_test_circumcenters_kernel(int* __restrict__ d_repair,
                                              const int3* __restrict__ d_circumcircles,
                                              const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int Nccs,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              int *anyFail
                                              )
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Nccs)
        return;

    //the indices of particles forming the circumcircle
    int3 i1 = d_circumcircles[idx];
    //the vertex we will take to be the origin, and its cell position
    double2 v = d_pt[i1.x];
    int ib=Floor(v.x/boxsize);
    int jb=Floor(v.y/boxsize);


    double2 pt1,pt2;
    Box.minDist(d_pt[i1.y],v,pt1);
    Box.minDist(d_pt[i1.z],v,pt2);

    //get the circumcircle
    double2 Q;
    double rad;
    Circumcircle(pt1,pt2,Q,rad);

    //look through cells for other particles...re-use pt1 and pt2 variables below
    bool badParticle = false;
    int wcheck = Ceil(rad/boxsize);

    if(wcheck > xsize/2) wcheck = xsize/2;
    rad = rad*rad;
    for (int ii = ib-wcheck; ii <= ib+wcheck; ++ii)
        {
        for (int jj = jb-wcheck; jj <= jb+wcheck; ++jj)
            {
            int cx = ii;
            if(cx < 0) cx += xsize;
            if(cx >= xsize) cx -= xsize;
            int cy = jj;
            if(cy < 0) cy += ysize;
            if(cy >= ysize) cy -= ysize;

            int bin = ci(cx,cy);

            for (int pp = 0; pp < d_cell_sizes[bin]; ++pp)
                {
                int newidx = d_cell_idx[cli(pp,bin)];

                Box.minDist(d_pt[newidx],v,pt1);
                Box.minDist(pt1,Q,pt2);

                //if it's in the circumcircle, check that its not one of the three points
                if(pt2.x*pt2.x+pt2.y*pt2.y < rad)
                    {
                    if (newidx != i1.x && newidx != i1.y && newidx !=i1.z)
                        {
                        badParticle = true;
                        d_repair[newidx] = 1;
                        };
                    };
                };//end loop over particles in the given cell
            };
        };// end loop over cells

    if (badParticle)
        {
        *anyFail = 1;
        d_repair[i1.x] = 1;
        d_repair[i1.y] = 1;
        d_repair[i1.z] = 1;
        };

    return;
    };


__host__ __device__ void computeVoronoiGeometryFunction(int idx,
                                          const double2* __restrict__ d_points,
                                          double2* __restrict__ d_AP,
                                          const int* __restrict__ d_nn,
                                          const int* __restrict__ d_n,
                                          double2* __restrict__ d_vc,
                                          double4* __restrict__ d_vln,
                                          Index2D n_idx,
                                          periodicBoundaries Box
                                          )
    {
    double2  nnextp, nlastp,pi,rij,rik,vlast,vnext,vfirst;

    int neigh = d_nn[idx];
    double Varea = 0.0;
    double Vperi= 0.0;

    pi = d_points[idx];
    nlastp = d_points[ d_n[n_idx(neigh-1,idx)] ];
    nnextp = d_points[ d_n[n_idx(0,idx)] ];

    Box.minDist(nlastp,pi,rij);
    Box.minDist(nnextp,pi,rik);
    Circumcenter(rij,rik,vfirst);
    vlast = vfirst;

    //set the VoroCur to this voronoi vertex
    //the convention is that nn=0 in this routine should be nn = 1 in the force sets calculation
    d_vc[n_idx(1,idx)] = vlast;

    for (int nn = 1; nn < neigh; ++nn)
        {
        rij = rik;
        int nid = d_n[n_idx(nn,idx)];
        nnextp = d_points[ nid ];
        Box.minDist(nnextp,pi,rik);
        Circumcenter(rij,rik,vnext);

        //fill in the VoroCur structure

        int idc = n_idx(nn+1,idx);
        if(nn == neigh-1)
            idc = n_idx(0,idx);

        d_vc[idc]=vnext;

        //...and back to computing the geometry
        Varea += TriangleArea(vlast,vnext);
        double dx = vlast.x - vnext.x;
        double dy = vlast.y - vnext.y;
        Vperi += sqrt(dx*dx+dy*dy);
        vlast=vnext;
        };
    Varea += TriangleArea(vlast,vfirst);
    double dx = vlast.x - vfirst.x;
    double dy = vlast.y - vfirst.y;
    Vperi += sqrt(dx*dx+dy*dy);

    //it's more memory-access friendly to now fill in the VoroLastNext structure separately
    vlast = d_vc[n_idx(neigh-1,idx)];
    vfirst = d_vc[n_idx(0,idx)];
    for (int nn = 0; nn < neigh; ++nn)
        {
        int idn = n_idx(nn+1,idx);
        if(nn == neigh-1) idn = n_idx(0,idx);
        vnext = d_vc[idn];

        int idc = n_idx(nn,idx);
        d_vln[idc].x = vlast.x;
        d_vln[idc].y = vlast.y;
        d_vln[idc].z = vnext.x;
        d_vln[idc].w = vnext.y;

        vlast = vfirst;
        vfirst = vnext;
        };

    d_AP[idx].x=Varea;
    d_AP[idx].y=Vperi;
    return;
    }
/*!
  Since the cells are guaranteed to be convex, the area of the cell is the sum of the areas of
  the triangles formed by consecutive Voronoi vertices
  */
__global__ void gpu_compute_voronoi_geometry_kernel(const double2* __restrict__ d_points,
                                          double2* __restrict__ d_AP,
                                          const int* __restrict__ d_nn,
                                          const int* __restrict__ d_n,
                                          double2* __restrict__ d_vc,
                                          double4* __restrict__ d_vln,
                                          int N,
                                          Index2D n_idx,
                                          periodicBoundaries Box
                                        )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    computeVoronoiGeometryFunction(idx,d_points,d_AP,d_nn,d_n,d_vc,d_vln, n_idx,Box);
    return;
    };



//!call the kernel to test every circumcenter to see if it's empty
bool gpu_test_circumcenters(int *d_repair,
                            int3 *d_ccs,
                            int Nccs,
                            double2 *d_pt,
                            unsigned int *d_cell_sizes,
                            int *d_idx,
                            int Np,
                            int xsize,
                            int ysize,
                            double boxsize,
                            periodicBoundaries &Box,
                            Index2D &ci,
                            Index2D &cli,
                            int *fail)
    {
    unsigned int block_size = 128;
    if (Nccs < 128) block_size = 32;
    unsigned int nblocks  = Nccs/block_size + 1;

    gpu_test_circumcenters_kernel<<<nblocks,block_size>>>(
                            d_repair,
                            d_ccs,
                            d_pt,
                            d_cell_sizes,
                            d_idx,
                            Nccs,
                            xsize,
                            ysize,
                            boxsize,
                            Box,
                            ci,
                            cli,
                            fail
                            );

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

//!Call the kernel to compute the geometry
bool gpu_compute_voronoi_geometry(const double2 *d_points,
                        double2   *d_AP,
                        const int      *d_nn,
                        const int      *d_n,
                        double2 *d_vc,
                        double4 *d_vln,
                        int      N,
                        Index2D  &n_idx,
                        periodicBoundaries &Box,
                        bool useGPU,
                        int nThreads
                        )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    if(useGPU)
        {
        gpu_compute_voronoi_geometry_kernel<<<nblocks,block_size>>>(                                        d_points,
                        d_AP,
                        d_nn,
                        d_n,
                        d_vc,
                        d_vln,
                        N,
                        n_idx,
                        Box
                        );
        HANDLE_ERROR(cudaGetLastError());
        return cudaSuccess;
        }
    else
        ompFunctionLoop(nThreads,N,computeVoronoiGeometryFunction,d_points,d_AP,d_nn,d_n,d_vc,d_vln,n_idx,Box);
        
        
    return true;
    };

__global__ void gpu_update_neighIdxs_kernel(int *neighborNum,
                          int *neighNumScan,
                          int2 *neighIdxs,
                          int Ncells)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Ncells)
        return;
    int nmax = neighborNum[idx];
    int offset = neighNumScan[idx];
    for (int ii = 0; ii < nmax; ++ii)
        {
        neighIdxs[offset+ii].x = idx;
        neighIdxs[offset+ii].y = ii;
        }
    return;
    }


bool gpu_update_neighIdxs(int *neighborNum,
                          int *neighNumScan,
                          int2 *neighIdxs,
                          int &NeighIdxNum,
                          int Ncells)
    {
    unsigned int block_size = 128;
    if (Ncells < 128) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;


    {
    thrust::device_ptr<int> dpNN(neighborNum);
    thrust::device_ptr<int> dpNNS(neighNumScan);
    thrust::exclusive_scan(dpNN,dpNN+Ncells,dpNNS);
    }

    gpu_update_neighIdxs_kernel<<<nblocks,block_size>>>(neighborNum,neighNumScan,neighIdxs,Ncells);

    {
    thrust::device_ptr<int> dpNN(neighborNum);
    NeighIdxNum = thrust::reduce(dpNN,dpNN+Ncells);//neighborNum,neighborNum+Ncells);
    }
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

__global__ void gpu_all_del_sets_kernel(int *neighborNum,
                      int *neighbors,
                      int2 *delSets,
                      int * delOther,
                      int Ncells,
                      Index2D nIdx)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Ncells)
        return;

    int iNeighs = neighborNum[idx];
    int nm1,n1,n2,nextNeighs,testPoint;
    nm1 = neighbors[nIdx(iNeighs-2,idx)];
    n1 = neighbors[nIdx(iNeighs-1,idx)];
    for(int nn = 0; nn < iNeighs; ++nn)
        {
        n2 = neighbors[nIdx(nn,idx)];
        nextNeighs = neighborNum[n1];
        for (int nn2 = 0; nn2 <nextNeighs; ++nn2)
            {
            testPoint = neighbors[nIdx(nn2,n1)];
            if(testPoint==nm1)
                {
                delOther[nIdx(nn,idx)] = neighbors[nIdx((nn2+1)%nextNeighs,n1)];
                break;
                }
            }
        delSets[nIdx(nn,idx)].x = nm1;
        delSets[nIdx(nn,idx)].y = n1;

        nm1=n1;
        n1=n2;
        }
    }

bool gpu_all_del_sets(int *neighborNum,
                      int *neighbors,
                      int2 *delSets,
                      int * delOther,
                      int Ncells,
                      Index2D &nIdx)
    {
    unsigned int block_size = 128;
    if (Ncells < 128) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;

    gpu_all_del_sets_kernel<<<nblocks,block_size>>>(neighborNum,neighbors,delSets,delOther, Ncells,nIdx);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
