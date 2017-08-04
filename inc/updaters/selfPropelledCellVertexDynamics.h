#ifndef selfPropelledCellVertexDynamics_H
#define selfPropelledCellVertexDynamics_H

#include "selfPropelledParticleDynamics.h"

/*! \file selfPropelledCellVertexDynamics.h */
//!A class that implements simple self-propelled particle dynamics in 2D
/*!
implements dr/dt = mu*F = 1/3 Sum(v_i \hat{n}_i, where \hat{n}_i = (cos(theta),sin(theta)), and
d theta/dt = (brownian noise), and the sum is over the cells that neighbor the vertex
*/
class selfPropelledCellVertexDynamics : public selfPropelledParticleDynamics
    {
    public:
        //!base constructor sets default time step size
        selfPropelledCellVertexDynamics(int Ncells,int Nvertices);

        //!the fundamental function that models will call, using vectors of different data structures
        virtual void integrateEquationsOfMotion();
        //!call the CPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionCPU();
        //!call the GPU routine to integrate the e.o.m.
        virtual void integrateEquationsOfMotionGPU();

        //! In the mixed cell-vertex model, the equations of motion need to know the number of cells
        int Ncells;
        //! In the mixed cell-vertex model, the equations of motion need to know the number of vertices (dof)
        int Nvertices;

        //!Get the number of vertices
        int getNvertices(){return Nvertices;};
        //!Get the number of cells
        int getNcells(){return Ncells;};
        //!Set the number of cells
        void setNcells(int _n){Ncells = _n;};
        //!Set the number of vertices
        void setNvertices(int _n){Nvertices = _n;Ndof = _n;};
    };
#endif
