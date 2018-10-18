#define ENABLE_CUDA

#include "selfPropelledVicsekAligningParticleDynamics.h"
#include "selfPropelledVicsekAligningParticleDynamics.cuh"
/*! \file selfPropelledVicsekAligningParticleDynamics.cpp */

/*!
An extremely simple constructor that does nothing, but enforces default GPU operation
\param the number of points in the system (cells or particles)
*/
selfPropelledVicsekAligningParticleDynamics::selfPropelledVicsekAligningParticleDynamics(int _N)
    {
    Timestep = 0;
    deltaT = 0.01;
    GPUcompute = false;
    mu = 1.0;
    Eta=0.0;
    Ndof = _N;
    noise.initialize(Ndof);
    displacements.resize(Ndof);
    };

/*!
When spatial sorting is performed, re-index the array of cuda RNGs... This function is currently
commented out, for greater flexibility (i.e., to not require that the indexToTag (or Itt) be the
re-indexing array), since that assumes cell and not particle-based dynamics
*/
void selfPropelledVicsekAligningParticleDynamics::spatialSorting()
    {
    //reIndexing = activeModel->returnItt();
    //reIndexRNG(noise.RNGs);
    };

/*!
Set the shared pointer of the base class to passed variable; cast it as an active cell model
*/
void selfPropelledVicsekAligningParticleDynamics::set2DModel(shared_ptr<Simple2DModel> _model)
    {
    model=_model;
    activeModel = dynamic_pointer_cast<Simple2DActiveCell>(model);
    }

/*!
Advances self-propelled dynamics with random noise in the director by one time step
*/
void selfPropelledVicsekAligningParticleDynamics::integrateEquationsOfMotion()
    {
    Timestep += 1;
    if (activeModel->getNumberOfDegreesOfFreedom() != Ndof)
        {
        Ndof = activeModel->getNumberOfDegreesOfFreedom();
        displacements.resize(Ndof);
        noise.initialize(Ndof);
        };
    if(GPUcompute)
        {
        integrateEquationsOfMotionGPU();
        }
    else
        {
        integrateEquationsOfMotionCPU();
        }
    }

/*!
The straightforward CPU implementation
*/
void selfPropelledVicsekAligningParticleDynamics::integrateEquationsOfMotionCPU()
    {
    activeModel->computeForces();
    {//scope for array Handles
    ArrayHandle<Dscalar2> h_f(activeModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<Dscalar> h_cd(activeModel->cellDirectors);
    ArrayHandle<Dscalar2> h_v(activeModel->cellVelocities);
    ArrayHandle<Dscalar2> h_disp(displacements,access_location::host,access_mode::overwrite);
    ArrayHandle<Dscalar2> h_motility(activeModel->Motility,access_location::host,access_mode::read);
    ArrayHandle<int> h_nn(activeModel->cellNeighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(activeModel->cellNeighbors,access_location::host,access_mode::read);

    for (int ii = 0; ii < Ndof; ++ii)
        {
        //displace according to current velocities and forces
        Dscalar theta = h_cd.data[ii];
        if(theta < -PI)
            theta += 2*PI;
        if(theta > PI)
            theta -= 2*PI;

        Dscalar v0i = h_motility.data[ii].x;
        Dscalar Dri = h_motility.data[ii].y;
        h_v.data[ii].x = (v0i * cos(theta) + mu * h_f.data[ii].x);
        h_v.data[ii].y = (v0i * sin(theta) + mu * h_f.data[ii].y);
        h_disp.data[ii] = deltaT*h_v.data[ii];
        h_v.data[ii] = h_disp.data[ii];

        //calculate the average direction of the neighbors
        Dscalar2 direction = make_Dscalar2(0.0,0.0);
        int neigh = h_nn.data[ii];
        for (int nn = 0; nn < neigh; ++nn)
            {
            Dscalar curTheta = h_cd.data[h_n.data[activeModel->n_idx(nn,ii)]];
            direction.x += cos(curTheta);
            direction.y += sin(curTheta);
            }
        Dscalar randomNumber = noise.getRealUniform(0.0,2*PI);
        direction.x += neigh*Eta*cos(randomNumber); 
        direction.y += neigh*Eta*sin(randomNumber); 

        Dscalar phi = atan2(direction.y,direction.x);

        //change the velocity vector to this new direction
        h_cd.data[ii] = phi; //theta+ randomNumber*sqrt(2.0*deltaT*Dri) - deltaT*J*sin(theta-phi);

        };
    }//end array handle scoping
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    //vector of displacements is mu*forces*timestep + v0's*timestep
    };

/*!
The straightforward GPU implementation
*/
void selfPropelledVicsekAligningParticleDynamics::integrateEquationsOfMotionGPU()
    {
    activeModel->computeForces();
    {//scope for array Handles
    ArrayHandle<Dscalar2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<Dscalar> d_cd(activeModel->cellDirectors,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_v(activeModel->cellVelocities,access_location::device,access_mode::readwrite);
    ArrayHandle<Dscalar2> d_disp(displacements,access_location::device,access_mode::overwrite);
    ArrayHandle<Dscalar2> d_motility(activeModel->Motility,access_location::device,access_mode::read);
    ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);
    ArrayHandle<int> d_nn(activeModel->cellNeighborNum,access_location::device,access_mode::read);
    ArrayHandle<int> d_n(activeModel->cellNeighbors,access_location::device,access_mode::read);

    gpu_spp_vicsek_aligning_eom_integration(d_f.data,
                 d_v.data,
                 d_disp.data,
                 d_motility.data,
                 d_cd.data,
                 d_nn.data,
                 d_n.data,
                 activeModel->n_idx,
                 d_RNG.data,
                 Ndof,
                 deltaT,
                 Timestep,
                 mu,
                 Eta);
    };//end array handle scope
    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    };
