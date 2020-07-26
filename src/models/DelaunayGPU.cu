#include <cuda_runtime.h>
#include "cellListGPU.cuh"
#include "indexer.h"
#include "periodicBoundaries.h"
#include "functions.h"
#include <iostream>
#include <stdio.h>
#include "DelaunayGPU.cuh"

/*! \file DelaunayGPU.cu */
/*!
    \addtogroup DelaunayGPUBaseKernels
    @{
*/

#define THREADCOUNT 128


__host__ __device__ inline double checkCCW(const double2 pa, const double2 pb, const double2 pc)
    {
    return (pa.x - pb.x) * (pa.y - pc.y) - (pa.y - pb.y) * (pa.x - pc.x);
    }
__host__ __device__ inline int checkCW(const double pax, const double pay, const double pbx, const double pby, const double pcx, const double pcy)
    {
    return ((pax - pbx) * (pay - pcy) - (pay - pby) * (pax - pcx)) >0 ? 0 : 1;
    }
__host__ __device__ inline unsigned positiveModulo(int i, unsigned n)
    {
    int mod = i % (int) n;
    if(i < 0) mod += n;
    return mod;
    };

template<typename T, int N = -1>
__device__ inline void rotateInMemoryRight( T *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    for (int ii = rotationSize+rotationOffset; ii > rotationOffset; ii--)
        {
        inList[saveIdx+ii+1] = inList[saveIdx+ii];
        }
    }
template<>
__device__ inline void rotateInMemoryRight<double2,1>( double2 *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    inList[saveIdx+rotationSize+rotationOffset+1] = inList[saveIdx+rotationSize+rotationOffset];
    };
template<>
__device__ inline void rotateInMemoryRight<double,1>(double *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    inList[saveIdx+rotationSize+rotationOffset+1] = inList[saveIdx+rotationSize+rotationOffset];
    };
template<>
__device__ inline void rotateInMemoryRight<int,1>( int *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    inList[saveIdx+rotationSize+rotationOffset+1] = inList[saveIdx+rotationSize+rotationOffset];
    };
template<>
__device__ inline void rotateInMemoryRight<double2,2>( double2 *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    double2 temp1,temp2;
    temp1 = inList[saveIdx+rotationSize+rotationOffset-1];
    temp2 = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset] = temp1;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp2;
    };
template<>
__device__ inline void rotateInMemoryRight<double,2>(double *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    double2 temp;
    temp.x = inList[saveIdx+rotationSize+rotationOffset-1];
    temp.y = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset] = temp.x;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp.y;
    };

template<>
__device__ inline void rotateInMemoryRight<int,2>( int *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    int2 temp;
    temp.x = inList[saveIdx+rotationSize+rotationOffset-1];
    temp.y = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset] = temp.x;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp.y;
    };
template<>
__device__ inline void rotateInMemoryRight<double2,3>( double2 *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    double2 temp1,temp2,temp3;
    temp1 = inList[saveIdx+rotationSize+rotationOffset-2];
    temp2 = inList[saveIdx+rotationSize+rotationOffset-1];
    temp3 = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset-1] = temp1;
    inList[saveIdx+rotationSize+rotationOffset] = temp2;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp3;
    };
template<>
__device__ inline void rotateInMemoryRight<double,3>(double *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    double3 temp;
    temp.x = inList[saveIdx+rotationSize+rotationOffset-2];
    temp.y = inList[saveIdx+rotationSize+rotationOffset-1];
    temp.z = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset-1] = temp.x;
    inList[saveIdx+rotationSize+rotationOffset] = temp.y;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp.z;
    };

template<>
__device__ inline void rotateInMemoryRight<int,3>( int *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    int3 temp;
    temp.x = inList[saveIdx+rotationSize+rotationOffset-2];
    temp.y = inList[saveIdx+rotationSize+rotationOffset-1];
    temp.z = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset-1] = temp.x;
    inList[saveIdx+rotationSize+rotationOffset] = temp.y;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp.z;
    };
template<>
__device__ inline void rotateInMemoryRight<double2,4>( double2 *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    double2 temp1,temp2,temp3,temp4;
    temp1 = inList[saveIdx+rotationSize+rotationOffset-3];
    temp2 = inList[saveIdx+rotationSize+rotationOffset-2];
    temp3 = inList[saveIdx+rotationSize+rotationOffset-1];
    temp4 = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset-2] = temp1;
    inList[saveIdx+rotationSize+rotationOffset-1] = temp2;
    inList[saveIdx+rotationSize+rotationOffset] = temp3;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp4;
    };
template<>
__device__ inline void rotateInMemoryRight<double,4>(double *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    double4 temp;
    temp.x = inList[saveIdx+rotationSize+rotationOffset-3];
    temp.y = inList[saveIdx+rotationSize+rotationOffset-2];
    temp.z = inList[saveIdx+rotationSize+rotationOffset-1];
    temp.w = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset-2] = temp.x;
    inList[saveIdx+rotationSize+rotationOffset-1] = temp.y;
    inList[saveIdx+rotationSize+rotationOffset] = temp.z;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp.w;
    };
template<>
__device__ inline void rotateInMemoryRight<int,4>( int *inList, int saveIdx, int rotationOffset,int rotationSize)
    {
    int4 temp;
    temp.x = inList[saveIdx+rotationSize+rotationOffset-3];
    temp.y = inList[saveIdx+rotationSize+rotationOffset-2];
    temp.z = inList[saveIdx+rotationSize+rotationOffset-1];
    temp.w = inList[saveIdx+rotationSize+rotationOffset];
    inList[saveIdx+rotationSize+rotationOffset-2] = temp.x;
    inList[saveIdx+rotationSize+rotationOffset-1] = temp.y;
    inList[saveIdx+rotationSize+rotationOffset] = temp.z;
    inList[saveIdx+rotationSize+rotationOffset+1] = temp.w;
    };

/*!
   Is a given cell bucket inside a given edge's angle?
*/
__device__ inline bool cellBucketInsideAngle(const double2 v, const int cx, const int cy, const double2 v1, const double2 v2, const double cellSize, periodicBoundaries &box)
    {
    double2 p1,p2,p3,p4;
    double2 pt = make_double2(0.,0.);
    double2 c1 = make_double2(cx*cellSize,cy*cellSize);
    double2 c2 = make_double2((cx+1)*cellSize,cy*cellSize);
    double2 c3 = make_double2((cx+1)*cellSize,(cy+1)*cellSize);
    double2 c4 = make_double2(cx*cellSize,(cy+1)*cellSize);

    box.minDist(v,c1,p1);
    if(checkCCW(pt, v1, p1)>0 && checkCCW(pt, v2, p1)<0)return true;
    box.minDist(v,c2,p2);
    if(checkCCW(pt, v1, p2)>0 && checkCCW(pt, v2, p2)<0)return true; 
    box.minDist(v,c3,p3);
    if(checkCCW(pt, v1, p3)>0 && checkCCW(pt, v2, p3)<0)return true; 
    box.minDist(v,c4,p4);
    if(checkCCW(pt, v1, p4)>0 && checkCCW(pt, v2, p4)<0)return true; 

    return false;
    }

//per-circumcircle test function
__host__ __device__ void test_circumcircle_kernel_function(int idx,
                                              int* __restrict__ d_repair,
                                              const int3* __restrict__ d_circumcircles,
                                              const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli
                                              )
    {
    //the indices of particles forming the circumcircle
    int3 i1 = d_circumcircles[idx];
    //the vertex we will take to be the origin, and its cell position
    double2 v = d_pt[i1.x];
    int ib=floor(v.x/boxsize);
    int jb=floor(v.y/boxsize);

    double2 pt1,pt2;
    Box.minDist(d_pt[i1.y],v,pt1);
    Box.minDist(d_pt[i1.z],v,pt2);


    //get the circumcircle
    double2 Q;
    double rad;
    Circumcircle(pt1,pt2,Q,rad);

    //look through cells for other particles...re-use pt1 and pt2 variables below
    bool badParticle = false;
    int wcheck = ceil(rad/boxsize)+1;

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
                if(newidx == i1.x || newidx == i1.y || newidx == i1.z)
                    continue;

                Box.minDist(d_pt[newidx],v,pt1);
                Box.minDist(pt1,Q,pt2);

                //if it's in the circumcircle, check that its not one of the three points
                if(pt2.x*pt2.x+pt2.y*pt2.y < rad)
                    {
                    d_repair[newidx] = newidx;
                    badParticle = true;
                    };
                if(badParticle) break;
                };//end loop over particles in the given cell
            if(badParticle) break;
            };
        if(badParticle) break;
        };// end loop over cells
    if (badParticle)
        {
          d_repair[i1.x] = i1.x;
          d_repair[i1.y] = i1.y;
          d_repair[i1.z] = i1.z;
        };
    return;
    }
/*!
  Independently check every triangle in the Delaunay mesh to see if the cirumcircle defined by the
  vertices of that triangle is empty. Use the cell list to ensure that only checks of nearby
  particles are required.
  */
__global__ void gpu_test_circumcircles_kernel(
                                              int* __restrict__ d_repair,
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
                                              Index2D cli
                                              )
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Nccs)
        return;
    test_circumcircle_kernel_function(idx,d_repair,d_circumcircles,d_pt,
                                      d_cell_sizes,d_cell_idx,xsize,ysize,
                                      boxsize,Box,ci,cli);

    return;
    };

/*!
device function carries out the task of finding a good enclosing polygon, using the virtual point and half-plane intersection method
*/
__host__ __device__ void virtual_voronoi_calc_function(        int kidx,
                                              const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries &Box,
                                              Index2D &ci,
                                              Index2D &cli,
                                              Index2D &GPU_idx
                                              )
    {
    unsigned int poly_size;
    int m, n;
    double2 pt1;
    double rr;
    double Lmax=(xsize*boxsize)*0.5; 
    double LL=Lmax/1.414213562373095-EPSILON;

    poly_size=4;
    P[GPU_idx(0, kidx)].x=LL;
    P[GPU_idx(0, kidx)].y=LL;
    P[GPU_idx(1, kidx)].x=-LL;
    P[GPU_idx(1, kidx)].y=LL;
    P[GPU_idx(2, kidx)].x=-LL;
    P[GPU_idx(2, kidx)].y=-LL;
    P[GPU_idx(3, kidx)].x=LL;
    P[GPU_idx(3, kidx)].y=-LL;
    /*
    poly_size=5;
    P[GPU_idx(0, kidx)].x=0.62*LL;
    P[GPU_idx(0, kidx)].y=1.9*LL;
    P[GPU_idx(1, kidx)].x=-1.62*LL;
    P[GPU_idx(1, kidx)].y=1.176*LL;
    P[GPU_idx(2, kidx)].x=-1.62*LL;
    P[GPU_idx(2, kidx)].y=-1.176*LL;
    P[GPU_idx(3, kidx)].x=.62*LL;
    P[GPU_idx(3, kidx)].y=-1.9*LL;
    P[GPU_idx(4, kidx)].x=2.0*LL;
    P[GPU_idx(4, kidx)].y=0.;
    */
    /*
    poly_size=6;
    P[GPU_idx(0, kidx)].x=2.*LL;
    P[GPU_idx(0, kidx)].y=0.;
    P[GPU_idx(1, kidx)].x=LL;
    P[GPU_idx(1, kidx)].y=1.7*LL;
    P[GPU_idx(2, kidx)].x=-LL;
    P[GPU_idx(2, kidx)].y=1.7*LL;
    P[GPU_idx(3, kidx)].x=-2.0*LL;
    P[GPU_idx(3, kidx)].y=0.;
    P[GPU_idx(4, kidx)].x=-LL;
    P[GPU_idx(4, kidx)].y=-1.7*LL;
    P[GPU_idx(5, kidx)].x=LL;
    P[GPU_idx(5, kidx)].y=-1.7*LL;
    */

#ifdef DEBUGFLAGUP
int blah = 0;
int blah2 = 0;
int blah3=0;
int maxCellsChecked=0;
int spotcheck=18;
int counter= 0 ;
if(kidx==spotcheck) printf("VP initial poly_size = %i\n",poly_size);
unsigned int t1,t2,t3,t4,t6,t7;
t6=0;
#endif

    for(m=0; m<poly_size; m++)
        {
        P_idx[GPU_idx(m, kidx)]=-1;
        n=m+1;
        if(n>=poly_size)n-=poly_size;
        Circumcircle(P[GPU_idx(m,kidx)],P[GPU_idx(n,kidx)], pt1, rr);
        Q[GPU_idx(m,kidx)]=pt1;
        Q_rad[GPU_idx(m,kidx)]=rr;
        }
#ifdef DEBUGFLAGUP
t1=clock();
#endif

    double2 disp, pt2, v;
    double xx, yy,currentRadius;
    unsigned int numberInCell, newidx, aa, removed, removeCW;
    int pp, w, j, jj, cx, cy, cc, dd, cell_rad, bin, cell_x, cell_y;


    v = d_pt[kidx];
    bool flag=false,removeCCW,firstRemove;

    int baseIdx = GPU_idx(0,kidx);
    for(jj=0; jj<poly_size; jj++)
        {
        currentRadius = Q_rad[GPU_idx(jj,kidx)];
        pt1=v;//+Q[GPU_idx(jj,kidx)]; //absolute position (within box) of circumcenter
        Box.putInBoxReal(pt1);
        currentRadius = Q_rad[GPU_idx(jj,kidx)];
        cell_x = (int)floor(pt1.x/boxsize) % xsize;
        cell_y = (int)floor(pt1.y/boxsize) % ysize;
        cell_rad = min((int) ceil(currentRadius/boxsize),xsize/2);
        cell_rad = (2*cell_rad+1);
        cc = 0;
        dd = 0;

        //check neighbours of Q's cell inside the circumcircle
#ifdef DEBUGFLAGUP
maxCellsChecked  = max(maxCellsChecked,cell_rad*cell_rad);
counter+=1;
t1=0;
t7=0;
t2=clock();
#endif
        for (int cellSpiral = 0; cellSpiral < cell_rad*cell_rad; ++cellSpiral)
            {
            cx = positiveModulo(cell_x+dd,xsize); 
            cy = positiveModulo(cell_y+cc,ysize); 

            //cue up the next pair of (dd,cc) cell indices relative to cell_x and cell_y
            if(abs(dd) <= abs(cc) && (dd != cc || dd >=0 ))
                {
                if (cc >=0)
                    dd += 1;
                else
                    dd -= 1;
                }
            else
                {
                if (dd >=0)
                    cc -= 1;
                else
                    cc += 1;
                }

                //check if there are any points in cellsns, if so do change, otherwise go for next bin
                bin = ci(cx,cy);
                numberInCell = d_cell_sizes[bin];

#ifdef DEBUGFLAGUP
//if(kidx==spotcheck) printf("(jj,ff) = (%i,%i)\t counter = %i \t cell_rad_in = %i \t cellIdex = %i\t numberInCell = %i\n", jj,ff,counter,cell_rad_in,bin,numberInCell);
#endif
                for (aa = 0; aa < numberInCell; ++aa)//check parts in cell
                    {
#ifdef DEBUGFLAGUP
blah +=1;
t3=clock();
#endif
                    newidx = d_cell_idx[cli(aa,bin)];
                    //6-Compute the half-plane Hv defined by the bissector of v and c, containing c
                    newidx = d_cell_idx[cli(aa,bin)];
                    if(newidx == kidx) continue;
                    bool skipPoint = false;
                    for (int pidx = 0; pidx < poly_size; ++pidx)
                        if(newidx == P_idx[GPU_idx(pidx, kidx)]) skipPoint = true;
                    if (skipPoint) continue;
#ifdef DEBUGFLAGUP
blah2+=1;
#endif
                    //how far is the point from the circumcircle's center?
                    rr=currentRadius*currentRadius;
                    Box.minDist(d_pt[newidx], v, disp); //disp = vector between new point and the point we're constructing the one ring of
                    Box.minDist(disp,Q[GPU_idx(jj, kidx)],pt1); // pt1 gets overwritten by vector between new point and Pi's circumcenter
                    if(pt1.x*pt1.x+pt1.y*pt1.y>rr)continue;
#ifdef DEBUGFLAGUP
blah3 +=1;
#endif
                    //calculate half-plane bissector
                    if(abs(disp.y)<THRESHOLD)
                        {
                        yy=disp.y/2+1;
                        xx=disp.x/2;
                        }
                    else if(abs(disp.x)<THRESHOLD)
                        {
                        yy=disp.y/2;
                        xx=disp.x/2+1;
                        }
                    else
                        {
                        yy=(disp.y*disp.y+disp.x*disp.x)/(2*disp.y);
                        xx=0;
                        }

                    //7-Q<-Hv intersect Q
                    //8-Update P, based on Q (Algorithm 2)      
                    cx = checkCW(0.5*disp.x,0.5*disp.y,xx,yy,0.,0.);
                    if(cx== checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[GPU_idx(jj,kidx)].x,Q[GPU_idx(jj,kidx)].y))
                        continue;
#ifdef DEBUGFLAGUP
t4=clock();
t1 += t4-t3;
#endif
                    //Remove the voronoi test points on the opposite half sector from the cell v
                    //If more than 1 voronoi test point is removed, then also adjust the delaunay neighbors of v
                    removeCW=0;
                    removeCCW=false;
                    firstRemove=true;
                    removed=0;
                    j=-1;
                    //which side will Q be at
                    cy = checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+poly_size-1].x,Q[baseIdx+poly_size-1].y);

                    removeCW=cy;
                    if(cy!=cx)
                        {
                        j=poly_size-1;
                        removed++;
                        removeCCW=true;
                        }

                    for(w=0; w<poly_size-1; w++)
                        {
                        cy = checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+w].x,Q[baseIdx+w].y);
                        if(cy!=cx)
                            {
                            if(removeCCW==false)
                                {
                                if(j<0)
                                    j=w;
                                else if(j>w)
                                    j=w;
                                removed++;
                                removeCCW=true;
                                }
                            else
                                {
                                if(firstRemove==false)
                                    {
                                    for(pp=w; pp<poly_size-1; pp++)
                                        {
                                        Q[baseIdx+pp]=Q[baseIdx+pp+1];
                                        }
                                    for(pp=w; pp<poly_size-1; pp++)
                                        {
                                        P[baseIdx+pp]=P[baseIdx+pp+1];
                                        }
                                    for(pp=w; pp<poly_size-1; pp++)
                                        {
                                        Q_rad[baseIdx+pp]=Q_rad[baseIdx+pp+1];
                                        }
                                    for(pp=w; pp<poly_size-1; pp++)
                                        {
                                        P_idx[baseIdx+pp]=P_idx[baseIdx+pp+1];
                                        }
                                    poly_size--;
                                    if(j>w)
                                        j--;
                                    w--;
                                    }
                                else 
                                    firstRemove=false;
                                removed++;
                                }	    
                            }
                            else
                                removeCCW=false;
                        }
                    if(removeCW!=cx && removeCCW==true && firstRemove==false)
                        {
                        poly_size--;
                        if(j>w)j--;
                        }

                    if(removed==0)
                        continue;

#ifdef DEBUGFLAGUP
t6 += clock()-t4;
t2=clock();
#endif
                    //Introduce new (if it exists) delaunay neighbor and new voronoi points
                    if(removed>1)
                        m=(j+2)%poly_size;
                    else 
                        m=(j+1)%poly_size;
                    Circumcircle(P[GPU_idx(j,kidx)], disp, pt1, xx);
                    Circumcircle(disp, P[GPU_idx(m,kidx)], pt2, yy);
                    if(removed==1)
                        {
                        poly_size++;
                        for(pp=poly_size-2; pp>j; pp--)
                            {
                            Q[GPU_idx(pp+1,kidx)]=Q[GPU_idx(pp,kidx)];
                            P[GPU_idx(pp+1,kidx)]=P[GPU_idx(pp,kidx)];
                            Q_rad[GPU_idx(pp+1,kidx)]=Q_rad[GPU_idx(pp,kidx)];
                            P_idx[GPU_idx(pp+1,kidx)]=P_idx[GPU_idx(pp,kidx)];
                            }
                        }

                    m=(j+1)%poly_size;
                    Q[GPU_idx(m,kidx)]=pt2;
                    Q_rad[GPU_idx(m,kidx)]=yy;
                    P[GPU_idx(m,kidx)]=disp;
                    P_idx[GPU_idx(m,kidx)]=newidx;

                    Q[GPU_idx(j,kidx)]=pt1;
                    Q_rad[GPU_idx(j,kidx)]=xx;
                    flag=true;
#ifdef DEBUGFLAGUP
t7 += clock()-t2;
#endif
                    break;
                    }//end checking all points in the current cell list cell
                if(flag==true)
                    break;
        }//end cell neighbor check, cell_rad_in
        if(flag==true)
            {
            flag=false;
            }
        }//end iterative loop over all edges of the 1-ring

    d_neighnum[kidx]=poly_size;
#ifdef DEBUGFLAGUP
    if(kidx==spotcheck)
        {
        printf("VP points checked for kidx %i = %i, ignore self points = %i, ignore points outside circumcircles = %i, total neighs = %i \n",kidx,blah,blah2,blah3,poly_size);
        printf("time checks : poly loading: %u \t subreplacement: %u \treplacement: %u\n",t1,t7,t6);
        };
#endif
    }

//assumes "fixlist" has the structure fixlist[ii]=-1 --> dont triangulate
__global__ void gpu_voronoi_calc_no_sort_kernel(const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              const int* __restrict__ d_fixlist,
                                              Index2D GPU_idx
                                              )
    {
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Ncells)return;
    if(d_fixlist[tidx] >= 0)
        {
        virtual_voronoi_calc_function(tidx,d_pt,d_cell_sizes,d_cell_idx,
                          P_idx, P, Q, Q_rad,
                          d_neighnum,
                          Ncells, xsize,ysize, boxsize,Box,
                          ci,cli,GPU_idx);
        };
    return;
    }

/*
GPU implementation of the DT. It makes use of a locallity lema described in (doi: 10.1109/ISVD.2012.9). It will only make the repair of the topology in case it is necessary. Steps are detailed as in paper.
*/
//This kernel constructs the initial test polygon.
//Currently it only uses 4 points, one in each quadrant.
//The initial test voronoi cell needs to be valid for the algorithm to work.
//Thus if the search fails, 4 virtual points are used at maximum distance as the starting polygon
__global__ void gpu_voronoi_calc_global_kernel(const double2* __restrict__ d_pt,
                                              const unsigned int* __restrict__ d_cell_sizes,
                                              const int* __restrict__ d_cell_idx,
                                              int* __restrict__ P_idx,
                                              double2* __restrict__ P,
                                              double2* __restrict__ Q,
                                              double* __restrict__ Q_rad,
                                              int* __restrict__ d_neighnum,
                                              int Ncells,
                                              int xsize,
                                              int ysize,
                                              double boxsize,
                                              periodicBoundaries Box,
                                              Index2D ci,
                                              Index2D cli,
                                              Index2D GPU_idx
                                              )
    {
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Ncells)return;

    virtual_voronoi_calc_function(tidx,d_pt,d_cell_sizes,d_cell_idx,
                          P_idx, P, Q, Q_rad,
                          d_neighnum,
                          Ncells, xsize,ysize, boxsize,Box,
                          ci,cli,GPU_idx);
    return;
    }

/*!
device function that goes from a candidate 1-ring to an actual 1-ring
*/
__host__ __device__ void get_oneRing_function(int kidx,
                const double2* __restrict__ d_pt,
                const unsigned int* __restrict__ d_cell_sizes,
                const int* __restrict__ d_cell_idx,
                int* __restrict__ P_idx,
                double2* __restrict__ P,
                double2* __restrict__ Q,
                double* __restrict__ Q_rad,
                int* __restrict__ d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries &Box,
                Index2D &ci,
                Index2D &cli,
                Index2D &GPU_idx,
                int const currentMaxNeighbors,
                int *maximumNeighborNumber
                )
    {
    //note that many of these variable names get re-used in different contexts throughout the kernel... take care
    double2 disp, pt1, pt2, v,currentQ;// v1, v2;
    double rr, xx, yy,currentRadius;
    unsigned int newidx, aa, removed, removeCW;
    int pp, m, w, j, jj, cx, cy, cc, dd, cell_rad, bin, cell_x, cell_y;

    v = d_pt[kidx];
    unsigned int poly_size=d_neighnum[kidx];
    bool flag=false, removeCCW, firstRemove;

    int baseIdx = GPU_idx(0,kidx);
    for(jj=0; jj<poly_size; jj++)
        {
        currentQ = Q[baseIdx+jj];
        pt1=v+currentQ; //absolute position (within box) of circumcenter
        //v1=P[GPU_idx(jj, kidx)];
        //v2=P[GPU_idx((jj+1)%poly_size, kidx)];
        Box.putInBoxReal(pt1);

        //check neighbours of Q's cell inside the circumcircle
        currentRadius = Q_rad[baseIdx+jj];
        cell_x = (int)floor(pt1.x/boxsize) % xsize;
        cell_y = (int)floor(pt1.y/boxsize) % ysize;
        cell_rad = min((int) ceil(currentRadius/boxsize),xsize/2);
        /*cells are currently checked in a spiral search from the central cell to the outermost...
        current algorithm searches CW, with the spiral being {{0,0},{1,0},{1,-1},...,{max,max}}.
        A small optimization could select the spiral used based on the quadrant relative to the
        base point
        */
        cell_rad = (2*cell_rad+1);
        cc = 0;
        dd = 0;
        for (int cellSpiral = 0; cellSpiral < cell_rad*cell_rad; ++cellSpiral)
            {
            cx = positiveModulo(cell_x+dd,xsize); 
            cy = positiveModulo(cell_y+cc,ysize); 

            //cue up the next pair of (dd,cc) cell indices relative to cell_x and cell_y
            if(abs(dd) <= abs(cc) && (dd != cc || dd >=0 ))
                {
                if (cc >=0)
                    dd += 1;
                else
                    dd -= 1;
                }
            else
                {
                if (dd >=0)
                    cc -= 1;
                else
                    cc += 1;
                }
            //if(cellBucketInsideAngle(v, cx, cy, v1, v2, boxsize, Box)==false)continue;

            //check if there are any points in cellsns, if so do change, otherwise go for next bin
            bin = ci(cx,cy);

            for(aa = 0; aa < d_cell_sizes[bin]; ++aa) //check points in cell
                {
                newidx = d_cell_idx[cli(aa,bin)];
                if(newidx == kidx || newidx == P_idx[baseIdx]) continue;
                bool skipPoint = false;
                for (int pidx = jj; pidx < poly_size; ++pidx)
                    if(newidx == P_idx[baseIdx+pidx]) skipPoint = true;
                if (skipPoint) continue;
                //6-Compute the half-plane Hv defined by the bissector of v and c, containing c
                //how far is the point from the circumcircle's center?
                rr=currentRadius*currentRadius;
                Box.minDist(d_pt[newidx], v, disp); //disp = vector between new point and the point we're constructing the one ring of
                Box.minDist(disp,currentQ,pt1); // pt1 gets overwritten by vector between new point and Pi's circumcenter
                if(pt1.x*pt1.x+pt1.y*pt1.y>rr)continue;
                //calculate half-plane bissector
                if(abs(disp.y) > THRESHOLD)
                    {
                    yy=(disp.y*disp.y+disp.x*disp.x)/(2*disp.y);
                    xx=0;
                    }
                else if(abs(disp.y)<THRESHOLD)
                    {
                    yy=disp.y/2+1;
                    xx=disp.x/2;
                    }
                if(abs(disp.x)<THRESHOLD)
                    {
                    yy=disp.y/2;
                    xx=disp.x/2+1;
                    }

                //7-Q<-Hv intersect Q
                //8-Update P, based on Q (Algorithm 2)      
                cx = checkCW(0.5*disp.x,0.5*disp.y,xx,yy,0.,0.);
                if(cx== checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+jj].x,Q[baseIdx+jj].y))
                    continue;

                //Remove the voronoi test points on the opposite half sector from the cell v
                //If more than 1 voronoi test point is removed, then also adjust the delaunay neighbors of v
                removeCW=0;
                removeCCW=false;
                firstRemove=true;
                removed=0;
                j=-1;
                //which side will Q be at
                cy = checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+poly_size-1].x,Q[baseIdx+poly_size-1].y);

                removeCW=cy;
                if(cy!=cx)
                    {
                    j=poly_size-1;
                    removed++;
                    if(jj == 0)
                        removeCCW=true;
                    }

                for(w=jj; w<poly_size-1; w++)
                    {
                    cy = checkCW(0.5*disp.x, 0.5*disp.y,xx,yy,Q[baseIdx+w].x,Q[baseIdx+w].y);
                    if(cy!=cx)
                        {
                        if(removeCCW==false)
                            {
                            if(j<0)
                                j=w;
                            else if(j>w)
                                j=w;
                            removed++;
                            removeCCW=true;
                            }
                        else
                            {
                            if(firstRemove==false)
                                {
                                for(pp=w; pp<poly_size-1; pp++)
                                    {
                                    Q[baseIdx+pp]=Q[baseIdx+pp+1];
                                    }
                                for(pp=w; pp<poly_size-1; pp++)
                                    {
                                    P[baseIdx+pp]=P[baseIdx+pp+1];
                                    }
                                for(pp=w; pp<poly_size-1; pp++)
                                    {
                                    Q_rad[baseIdx+pp]=Q_rad[baseIdx+pp+1];
                                    }
                                for(pp=w; pp<poly_size-1; pp++)
                                    {
                                    P_idx[baseIdx+pp]=P_idx[baseIdx+pp+1];
                                    }
                                poly_size--;
                                if(j>w)
                                    j--;
                                w--;
                                }
                            else 
                                firstRemove=false;
                            removed++;
                            }	    
                        }
                        else
                            removeCCW=false;
                    }
                if(removeCW!=cx && removeCCW==true && firstRemove==false)
                    {
                    poly_size--;
                    if(j>w)j--;
                    }

                if(removed==0)
                    continue;

                //Introduce new (if it exists) delaunay neighbor and new voronoi points
                if(removed>1)
                    m=(j+2)%poly_size;
                else 
                    m=(j+1)%poly_size;
                Circumcircle(P[baseIdx+j], disp, pt1, xx);
                Circumcircle(disp, P[baseIdx+m], pt2, yy);
                if(removed==1)
                    {
                    //if(kidx ==18 ) printf("kidx %i shifting poly by %i\n",kidx,poly_size-1-j);
                    poly_size++;
                    #ifdef __CUDA_ARCH__
                    if(poly_size > currentMaxNeighbors)
                        {
                        atomicMax(&maximumNeighborNumber[0],poly_size);
                        return;
                        }
                    int rotationSize = poly_size-2-j;
                    
                    switch(rotationSize)
                        {
                        case 0:
                            break;
                        case 1:
                            rotateInMemoryRight<double2, 1>(Q,baseIdx,j,rotationSize);
                            rotateInMemoryRight<double2, 1>(P,baseIdx,j,rotationSize);
                            rotateInMemoryRight<double,1>(Q_rad,baseIdx,j,rotationSize);
                            rotateInMemoryRight<int, 1>(P_idx,baseIdx,j,rotationSize);
                            break;
                        case 2:
                            rotateInMemoryRight<double2, 2>(Q,baseIdx,j,rotationSize);
                            rotateInMemoryRight<double2, 2>(P,baseIdx,j,rotationSize);
                            rotateInMemoryRight<double,2>(Q_rad,baseIdx,j,rotationSize);
                            rotateInMemoryRight<int, 2>(P_idx,baseIdx,j,rotationSize);
                            break;
                        case 3:
                            rotateInMemoryRight<double2, 3>(Q,baseIdx,j,rotationSize);
                            rotateInMemoryRight<double2, 3>(P,baseIdx,j,rotationSize);
                            rotateInMemoryRight<double,3>(Q_rad,baseIdx,j,rotationSize);
                            rotateInMemoryRight<int, 3>(P_idx,baseIdx,j,rotationSize);
                            break;
                        case 4:
                            rotateInMemoryRight<double2, 4>(Q,baseIdx,j,rotationSize);
                            rotateInMemoryRight<double2, 4>(P,baseIdx,j,rotationSize);
                            rotateInMemoryRight<double,4>(Q_rad,baseIdx,j,rotationSize);
                            rotateInMemoryRight<int, 4>(P_idx,baseIdx,j,rotationSize);
                            break;
                        default:
                            rotateInMemoryRight(Q,baseIdx,j,rotationSize);
                            rotateInMemoryRight(P,baseIdx,j,rotationSize);
                            rotateInMemoryRight(Q_rad,baseIdx,j,rotationSize);
                            rotateInMemoryRight(P_idx,baseIdx,j,rotationSize);
                        }
                    #else
                    for(pp=poly_size-2; pp>j; pp--)
                        {
                        Q[GPU_idx(pp+1,kidx)]=Q[GPU_idx(pp,kidx)];
                        P[GPU_idx(pp+1,kidx)]=P[GPU_idx(pp,kidx)];
                        Q_rad[GPU_idx(pp+1,kidx)]=Q_rad[GPU_idx(pp,kidx)];
                        P_idx[GPU_idx(pp+1,kidx)]=P_idx[GPU_idx(pp,kidx)];
                        }
                    #endif
                    }
                m=(j+1)%poly_size;
                Q[baseIdx+m]=pt2;
                Q[baseIdx+j]=pt1;
                Q_rad[baseIdx+m]=yy;
                Q_rad[baseIdx+j]=xx;

                P[baseIdx+m]=disp;
                P_idx[baseIdx+m]=newidx;

                flag=true;
                break;
                }//end checking all points in the current cell list cell
            if(flag==true)
                break;
            }//end spiral check
        if(flag==true)
            {
            jj--;
            flag=false;
            }
        }//end iterative loop over all edges of the 1-ring

    d_neighnum[kidx]=poly_size;

    return;
    }//end function

__global__ void gpu_get_neighbors_no_sort_kernel(const double2* __restrict__ d_pt,
                const unsigned int* __restrict__ d_cell_sizes,
                const int* __restrict__ d_cell_idx,
                int* __restrict__ P_idx,
                double2* __restrict__ P,
                double2* __restrict__ Q,
                double* __restrict__ Q_rad,
                int* __restrict__ d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                const int* __restrict__ d_fixlist,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum
                )
    {


    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Ncells)return;
    if(d_fixlist[tidx] <0)
        return;

    get_oneRing_function(tidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,maximumNeighborNum);

    return;
    }//end function

//This kernel updates the initial polygon into the real delaunay one.
//It goes through the same steps as in the paper, using the half plane intersection routine.
//It outputs the complete triangulation per point in CCW order
//!global get neighbors does not need a fixlist
__global__ void gpu_get_neighbors_global_kernel(const double2* __restrict__ d_pt,
                const unsigned int* __restrict__ d_cell_sizes,
                const int* __restrict__ d_cell_idx,
                int* __restrict__ P_idx,
                double2* __restrict__ P,
                double2* __restrict__ Q,
                double* __restrict__ Q_rad,
                int* __restrict__ d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum
                )
    {
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= Ncells)return;

    get_oneRing_function(tidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,maximumNeighborNum);
        
    return;
    }//end function

__global__ void gpu_get_circumcircles_kernel(int *neighbors, int *neighnum, int3 *ccs, int *assist, int N,Index2D GPU_idx)
    {
    unsigned int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= N)return;

    int nmax = neighnum[cell];
    int circumcircleIdx;
    int3 cc;
    cc.x=cell;
    cc.y = neighbors[GPU_idx(nmax-1,cell)];
    for (int jj = 0; jj < nmax; ++jj)
        {
        cc.z = neighbors[GPU_idx(jj,cell)];
        if(cc.x < cc.y && cc.x < cc.z)
            {
            circumcircleIdx = atomicAdd(&assist[0],1);
            ccs[circumcircleIdx] = cc;
            }
        cc.y = cc.z;
        }
    }

/////////////////////////////////////////////////////////////
//////
//////			Kernel Calls
//////
/////////////////////////////////////////////////////////////

bool gpu_voronoi_calc_no_sort(double2* d_pt,
                      unsigned int* d_cell_sizes,
                      int* d_cell_idx,
                      int* P_idx,
                      double2* P,
                      double2* Q,
                      double* Q_rad,
                      int* d_neighnum,
                      int Ncells,
                      int xsize,
                      int ysize,
                      double boxsize,
                      periodicBoundaries Box,
                      Index2D ci,
                      Index2D cli,
                      int* d_fixlist,
                      Index2D GPU_idx,
                      bool GPUcompute
                      )
    {
    unsigned int block_size = THREADCOUNT;
    if (Ncells < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;
    if(GPUcompute==true)
        {
        gpu_voronoi_calc_no_sort_kernel<<<nblocks,block_size>>>(
                        d_pt,
                        d_cell_sizes,
                        d_cell_idx,
                        P_idx,
                        P,
                        Q,
                        Q_rad,
                        d_neighnum,
                        Ncells,
                        xsize,
                        ysize,
                        boxsize,
                        Box,
                        ci,
                        cli,
                        d_fixlist,
                        GPU_idx
                        );
        HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
        cudaDeviceSynchronize();
#endif
        return cudaSuccess;
        }
    else
        {
        for(int tidx=0; tidx<Ncells; tidx++)
            {
            if(d_fixlist[tidx]>=0)
                virtual_voronoi_calc_function(tidx,d_pt,d_cell_sizes,d_cell_idx,
                      P_idx, P, Q, Q_rad,
                      d_neighnum,
                      Ncells, xsize,ysize, boxsize,Box,
                      ci,cli,GPU_idx);
            }
        }
    return true;
    }

bool gpu_voronoi_calc(double2* d_pt,
                unsigned int* d_cell_sizes,
                int* d_cell_idx,
                int* P_idx,
                double2* P,
                double2* Q,
                double* Q_rad,
                int* d_neighnum,
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                Index2D GPU_idx,
                bool GPUcompute
                )
{
    unsigned int block_size = THREADCOUNT;
    if (Ncells < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;

    if(GPUcompute==true)
        {
        gpu_voronoi_calc_global_kernel<<<nblocks,block_size>>>(
                        d_pt,
                        d_cell_sizes,
                        d_cell_idx,
                        P_idx,
                        P,
                        Q,
                        Q_rad,
                        d_neighnum,
                        Ncells,
                        xsize,
                        ysize,
                        boxsize,
                        Box,
                        ci,
                        cli,
                        GPU_idx
                        );

        HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP

        cudaDeviceSynchronize();
#endif
        return cudaSuccess;
        }
    else
        {
        for(int tidx=0; tidx<Ncells; tidx++)
             virtual_voronoi_calc_function(tidx,d_pt,d_cell_sizes,d_cell_idx,
                          P_idx, P, Q, Q_rad,
                          d_neighnum,
                          Ncells, xsize,ysize, boxsize,Box,
                          ci,cli,GPU_idx);
        }
    return true;
};

bool gpu_get_neighbors_no_sort(double2* d_pt, //the point set
                unsigned int* d_cell_sizes,//points per bucket
                int* d_cell_idx,//cellListIdxs
                int* P_idx,//index of Del Neighbors
                double2* P,//location del neighborPositions
                double2* Q,//voronoi vertex positions
                double* Q_rad,//radius? associated with voro vertex
                int* d_neighnum,//number of del neighbors
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                int* d_fixlist,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum,
                bool GPUcompute
                )
    {
    unsigned int block_size = THREADCOUNT;
    if (Ncells < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;
    if(GPUcompute==true)
        {
        gpu_get_neighbors_no_sort_kernel<<<nblocks,block_size>>>(
                      d_pt,d_cell_sizes,d_cell_idx,P_idx,P,Q,Q_rad,d_neighnum,Ncells,xsize,ysize,
                      boxsize,Box,ci,cli,d_fixlist,GPU_idx,maximumNeighborNum,currentMaxNeighborNum
                      );

        HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
        cudaDeviceSynchronize();
#endif
        return cudaSuccess;
        }
    else
        {
        for(int tidx=0; tidx<Ncells; tidx++)
            {
            if(d_fixlist[tidx]>=0)
                get_oneRing_function(tidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, 
                                 P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,
                                 boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,
                                 maximumNeighborNum);
            }
        }
    return true;
    };

bool gpu_get_neighbors(double2* d_pt, //the point set
                unsigned int* d_cell_sizes,//points per bucket
                int* d_cell_idx,//cellListIdxs
                int* P_idx,//index of Del Neighbors
                double2* P,//location del neighborPositions
                double2* Q,//voronoi vertex positions
                double* Q_rad,//radius? associated with voro vertex
                int* d_neighnum,//number of del neighbors
                int Ncells,
                int xsize,
                int ysize,
                double boxsize,
                periodicBoundaries Box,
                Index2D ci,
                Index2D cli,
                Index2D GPU_idx,
                int *maximumNeighborNum,
                int currentMaxNeighborNum,
                bool GPUcompute
                )
    {
    unsigned int block_size = THREADCOUNT;
    if (Ncells < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Ncells/block_size + 1;

    if(GPUcompute==true)
        {
        gpu_get_neighbors_global_kernel<<<nblocks,block_size>>>(
                      d_pt,d_cell_sizes,d_cell_idx,P_idx,P,Q,Q_rad,d_neighnum,
                      Ncells,xsize,ysize,boxsize,Box,ci,cli,GPU_idx,maximumNeighborNum,currentMaxNeighborNum
                      );

        HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
        cudaDeviceSynchronize();
#endif
        return cudaSuccess;
        }
    else
        {
        for(int tidx=0; tidx<Ncells; tidx++)
            get_oneRing_function(tidx, d_pt,d_cell_sizes,d_cell_idx,P_idx, 
                                 P,Q,Q_rad,d_neighnum, Ncells,xsize,ysize,
                                 boxsize,Box,ci,cli,GPU_idx, currentMaxNeighborNum,
                                 maximumNeighborNum);
        }
    return true;
    };

bool gpu_get_circumcircles(int *neighbors,
                           int *neighnum,
                           int3 *circumcircles,
                           int *assist,
                           int N,
                           Index2D &nIdx
                          )
    {
    unsigned int block_size = THREADCOUNT;
    if (N < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_get_circumcircles_kernel<<<nblocks,block_size>>>(
                            neighbors,
                            neighnum,
                            circumcircles,
                            assist,
                            N,
                            nIdx);
    HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
    cudaDeviceSynchronize();
#endif
    return cudaSuccess;
    }

//!call the kernel to test every circumcenter to see if it's empty
bool gpu_test_circumcircles(int *d_repair,
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
                            bool GPUcompute
                            )
    {
    unsigned int block_size = THREADCOUNT;
    if (Nccs < THREADCOUNT) block_size = 32;
    unsigned int nblocks  = Nccs/block_size + 1;

    if(GPUcompute)
        {
        gpu_test_circumcircles_kernel<<<nblocks,block_size>>>(
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
                            cli
                            );

        HANDLE_ERROR(cudaGetLastError());
#ifdef DEBUGFLAGUP
        cudaDeviceSynchronize();
#endif
        return cudaSuccess;
        }
    else
        {
        for(int idx = 0; idx < Nccs; ++idx)
            test_circumcircle_kernel_function(idx,d_repair,d_ccs,d_pt,
                                      d_cell_sizes,d_idx,xsize,ysize,
                                      boxsize,Box,ci,cli);
        }
    return true;
    };

/** @} */ //end of group declaration
