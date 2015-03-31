/***************************************************************************
 *   Copyright (C) 2014 by Fabíola Martins Campos de Oliveira              *
 *   fabiola.bass@gmail.com                                                *
 *   UNICAMP - Universidade Estadual de Campinas						   *
 *                       						   						   *
 *   This file is part of lattibol.					   					   *
 *                                                  			   		   *
 *   lattibol is free software: you can redistribute it and/or modify	   *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation, either version 3 of the License, or     *
 *   (at your option) any later version.				   				   *
 *									   									   *
 *   lattibol is distributed in the hope that it will be useful,	   	   *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of	       *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	       *
 *   GNU General Public License for more details.			   			   *
 *									   									   *
 *   You should have received a copy of the GNU General Public License     *
 *   long with lattibol.  If not, see <http://www.gnu.org/licenses/>.      *
 ***************************************************************************/

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_cuda_gl.h>    // CUDA device + OpenGL initialization functions

// Shared Library Test Functions
#include <helper_functions.h>  // CUDA SDK Helper functions

#include <mpi.h>


#include "simulation.h"
#include "simulation_d2q9.h"

#include "lattibol.h"
#include "input.h"
#include "output.h"

#include "collision.h"
#include "streaming.h"
#include "get_border.h"
#include "copy_border.h"
#include "apply_border.h"
#include "read_value.h"

#include <iostream>
using namespace std;

SimulationD2Q9::SimulationD2Q9(Input* inputPtr) : Simulation(inputPtr)
{
    data = new DataD2Q9[deviceCount];

	w0 = 16./36.;
	w1 = 4./36.;
	w2 = 1./36.;

    int i;
	int xn, x, xp, yn, y, yp;

	// Create variables on host side
    float *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8;
    float *ux, *uy, *rho;
	bool *solid;
    bool *solid_bound;

    // External forces
    float Fx, Fy;
    //float ueq_x, ueq_y;
    float tauInv;

    // Multiphase fluid (interaction potential)
    float *psi;
    float ux_aux, uy_aux, rho_aux;
    float *ueq_x, *ueq_y;

	// Allocate memories on host side
    ux = new float[size];
    uy = new float[size];
    rho = new float[size];
    solid_bound = new bool[size];
    f0 = new float[size];
    f1 = new float[size];
    f2 = new float[size];
    f3 = new float[size];
    f4 = new float[size];
    f5 = new float[size];
    f6 = new float[size];
    f7 = new float[size];
    f8 = new float[size];
    solid = new bool[size];
    
    // Multiphase fluid
	psi = new float[size];
	ueq_x = new float[size];
	ueq_y = new float[size];

   	for(i=0; i < size; i++) solid_bound[i]=0;

	bool *allSolids = input->getSolids();

	ux[0] = input->getUx();
    uy[0] = input->getUy();
    rho[0] = input->getRho();

	ueq_x[0] = input->getUx();
	ueq_y[0] = input->getUy();
	
	// External forces
	if (input->getExtFx() != 0)
	{
        tauInv = input->getTauInv();
        Fx = input->getExtFx();
		ueq_x[0] = ux[0] + Fx / (tauInv * rho[0]);
	}
	if (input->getExtFy() != 0)
	{
        tauInv = input->getTauInv();
        Fy = input->getExtFy();
		ueq_y[0] = uy[0] + Fy / (tauInv * rho[0]);
	}

   	// Interaction force (multiphase fluid)
   	if (input->getG() != 0)
   	{
   		float IFx, IFy;
		int offset[9];
		int width = input->getWidth();
		int height = input->getHeight();
		float G = input->getG();

		tauInv = input->getTauInv();
	    rho_aux = input->getRho();

	    srandom(0);
   		for (y = 0; y < height; y++)
			for (x = 0; x < width; x++)
			{
				offset[0] = y * width + x;
				if (!allSolids[offset[0]])
				{
					// Vertical flat interface
					/*if (x < width / 8 )
						rho[offset[0]] = 80.;
					else
						rho[offset[0]] = 500.;*/

					// Horizontal flat interface
					/*if (offset[0] < (width * height) / 2 )
						rho[offset[0]] = 80.;
					else
						rho[offset[0]] = 500.;*/

					// Seed bubbles
					if (x > 100 && x < 146 && y > 100 && y < 146)
						rho[offset[0]] = 85.7;
					else
						rho[offset[0]] = 524.39;

					// Capillary rise
					/*if (y < 180)
						rho[offset[0]] = 524.39;
					else
						rho[offset[0]] = 85.7;*/

					// Phase separation
					rho[offset[0]] = rho_aux;// + 0.005 * rho_aux * random() / ((double) RAND_MAX);

					psi[offset[0]] = 4. * exp(- 200. / rho[offset[0]]);
					ux[offset[0]] = ux[0];
					uy[offset[0]] = uy[0];
				}
			}

		for (y = 0; y < height; y++)
		{
			yp = (y < (height - 1)) ? (y + 1):(0);
			yn = (y > 0           ) ? (y - 1):(height - 1);
			for (x = 0; x < width; x++)
			{
				xp = (x < (width - 1)) ? (x + 1):(0);
				xn = (x > 0          ) ? (x - 1):(width - 1);

				offset[0] = y  * width + x;
				if (!allSolids[offset[0]])
				{
					offset[1] = y  * width + xp;
					offset[2] = yp * width + x;
					offset[3] = y  * width + xn;
					offset[4] = yn * width + x;
					offset[5] = yp * width + xp;
					offset[6] = yp * width + xn;
					offset[7] = yn * width + xn;
					offset[8] = yn * width + xp;

					IFx = 0;
					IFy = 0;

					IFx = IFx + w1 * psi[offset[1]];
					IFx = IFx - w1 * psi[offset[3]];
					IFx = IFx + w2 * psi[offset[5]];
					IFx = IFx - w2 * psi[offset[6]];
					IFx = IFx - w2 * psi[offset[7]];
					IFx = IFx + w2 * psi[offset[8]];

					IFy = IFy + w1 * psi[offset[2]];
					IFy = IFy - w1 * psi[offset[4]];
					IFy = IFy + w2 * psi[offset[5]];
					IFy = IFy + w2 * psi[offset[6]];
					IFy = IFy - w2 * psi[offset[7]];
					IFy = IFy - w2 * psi[offset[8]];

					IFx = - G * psi[offset[0]] * IFx;
					IFy = - G * psi[offset[0]] * IFy;

					// Para usar gravidade, coloque ueq_x[0] e ueq_y[0] no lugar de x[0] e y[0]
					ueq_x[offset[0]] = ueq_x[0] + (IFx / (tauInv * rho[offset[0]]));
					ueq_y[offset[0]] = ueq_y[0] + (IFy / (tauInv * rho[offset[0]]));
					// End interaction force

					// Fluid-Surface forces
					float Gads = input->getGads();
					float Fsx = 0;
					float Fsy = 0;

					if (allSolids[offset[1]])
						Fsx = Fsx + w1;
					if (allSolids[offset[2]])
						Fsy = Fsy - w1;
					if (allSolids[offset[3]])
						Fsx = Fsx - w1;
					if (allSolids[offset[4]])
						Fsy = Fsy + w1;
					if (allSolids[offset[5]])
					{
						Fsx = Fsx + w2;
						Fsy = Fsy - w2;
					}
					if (allSolids[offset[6]])
					{
						Fsx = Fsx - w2;
						Fsy = Fsy - w2;
					}
					if (allSolids[offset[7]])
					{
						Fsx = Fsx - w2;
						Fsy = Fsy + w2;
					}
					if (allSolids[offset[8]])
					{
						Fsx = Fsx + w2;
						Fsy = Fsy + w2;
					}

					Fsx = - Gads * psi[offset[0]] * Fsx;
					Fsy = - Gads * psi[offset[0]] * Fsy;

					ueq_x[offset[0]] = ueq_x[offset[0]] + (Fsx / (tauInv * rho[offset[0]]));
					ueq_y[offset[0]] = ueq_y[offset[0]] + (Fsy / (tauInv * rho[offset[0]]));

					f0[offset[0]] = w0*rho[offset[0]]*( 1.                                                              																  - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
					f1[offset[0]] = w1*rho[offset[0]]*( 1. + 3.*(+ueq_x[offset[0]]       			) + 4.5*(+ueq_x[offset[0]]       			 )*(+ueq_x[offset[0]]        			) - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
					f2[offset[0]] = w1*rho[offset[0]]*( 1. + 3.*(      			  +ueq_y[offset[0]] ) + 4.5*(      			   +ueq_y[offset[0]] )*(       			  +ueq_y[offset[0]] ) - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
					f3[offset[0]] = w1*rho[offset[0]]*( 1. + 3.*(-ueq_x[offset[0]]      			) + 4.5*(-ueq_x[offset[0]]     			     )*(-ueq_x[offset[0]]       			) - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
					f4[offset[0]] = w1*rho[offset[0]]*( 1. + 3.*(      			  -ueq_y[offset[0]] ) + 4.5*(       		   -ueq_y[offset[0]] )*(       			  -ueq_y[offset[0]] ) - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
					f5[offset[0]] = w2*rho[offset[0]]*( 1. + 3.*( ueq_x[offset[0]]+ueq_y[offset[0]] ) + 4.5*( ueq_x[offset[0]] +ueq_y[offset[0]] )*( ueq_x[offset[0]] +ueq_y[offset[0]] ) - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
					f6[offset[0]] = w2*rho[offset[0]]*( 1. + 3.*(-ueq_x[offset[0]]+ueq_y[offset[0]] ) + 4.5*(-ueq_x[offset[0]] +ueq_y[offset[0]] )*(-ueq_x[offset[0]] +ueq_y[offset[0]] ) - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
					f7[offset[0]] = w2*rho[offset[0]]*( 1. + 3.*(-ueq_x[offset[0]]-ueq_y[offset[0]] ) + 4.5*(-ueq_x[offset[0]] -ueq_y[offset[0]] )*(-ueq_x[offset[0]] -ueq_y[offset[0]] ) - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
					f8[offset[0]] = w2*rho[offset[0]]*( 1. + 3.*( ueq_x[offset[0]]-ueq_y[offset[0]] ) + 4.5*( ueq_x[offset[0]] -ueq_y[offset[0]] )*( ueq_x[offset[0]] -ueq_y[offset[0]] ) - 1.5*(ueq_x[offset[0]] * ueq_x[offset[0]] + ueq_y[offset[0]] * ueq_y[offset[0]] ) );
				}
			}
		}
   	} else
   	{
		f0[0] = w0*rho[0]*( 1.                                                                                - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );
		f1[0] = w1*rho[0]*( 1. + 3.*(+ueq_x[0]          ) + 4.5*(+ueq_x[0]          )*(+ueq_x[0]            ) - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );
		f2[0] = w1*rho[0]*( 1. + 3.*(         +ueq_y[0] ) + 4.5*(         +ueq_y[0] )*(         +ueq_y[0]   ) - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );
		f3[0] = w1*rho[0]*( 1. + 3.*(-ueq_x[0]          ) + 4.5*(-ueq_x[0]          )*(-ueq_x[0]            ) - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );
		f4[0] = w1*rho[0]*( 1. + 3.*(         -ueq_y[0] ) + 4.5*(         -ueq_y[0] )*(         -ueq_y[0]   ) - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );
		f5[0] = w2*rho[0]*( 1. + 3.*( ueq_x[0]+ueq_y[0] ) + 4.5*( ueq_x[0] +ueq_y[0] )*( ueq_x[0] +ueq_y[0] ) - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );
		f6[0] = w2*rho[0]*( 1. + 3.*(-ueq_x[0]+ueq_y[0] ) + 4.5*(-ueq_x[0] +ueq_y[0] )*(-ueq_x[0] +ueq_y[0] ) - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );
		f7[0] = w2*rho[0]*( 1. + 3.*(-ueq_x[0]-ueq_y[0] ) + 4.5*(-ueq_x[0] -ueq_y[0] )*(-ueq_x[0] -ueq_y[0] ) - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );
		f8[0] = w2*rho[0]*( 1. + 3.*( ueq_x[0]-ueq_y[0] ) + 4.5*( ueq_x[0] -ueq_y[0] )*( ueq_x[0] -ueq_y[0] ) - 1.5*(ueq_x[0] * ueq_x[0] + ueq_y[0] * ueq_y[0] ) );

		// Calculate initial density functions for initial velocity condition
		for (i=0; i < size; i++)
		{
			ux[i] = ux[0];
			uy[i] = uy[0];
			rho[i] = rho[0];
			f0[i] = f0[0];
			f1[i] = f1[0];
			f2[i] = f2[0];
			f3[i] = f3[0];
			f4[i] = f4[0];
			f5[i] = f5[0];
			f6[i] = f6[0];
			f7[i] = f7[0];
			f8[i] = f8[0];
		}
   	}

    for (i = 0; i < deviceCount; i++)
    {
    	//checkCudaErrors(cudaSetDevice(i));

    	data[i].width  = width;
    	data[i].height = height;
    	data[i].deviceCount = deviceCount;
    	data[i].dev = i;

    	// Get subdomain solids
    	for (y=0; y<height; y++) for (x=0; x<width; x++)
    		solid[x+y*width] = allSolids[(x + i*width)+y*width*deviceCount];

    	// If all boundaries are solid, solid_bound=1;
    	for (y=0; y<height; y++) for (x=1; x<width-1; x++)
    	{
    		xn = x-1;
    		xp = x+1;
    		// If y is the most top node, the top neighbor will be the most bottom node
    		yn = (y > 0         ) ? (y-1) : (height-1);
    		// If y is the most bottom node, the bottom neighbor will be the most top node
    		yp = (y < (height-1)) ? (y+1) : (0       );

    		if (
    				solid[x +y *width] && solid[xp+y *width] && solid[x +yp*width] &&
    				solid[xn+y *width] && solid[x +yn*width] && solid[xp+yp*width] &&
    				solid[xn+yp*width] && solid[xn+yn*width] && solid[xp+yn*width])
    			solid_bound[x+y*width] = 1;
    		else solid_bound[x+y*width] = 0;
    	}

        checkCudaErrors( cudaStreamCreate(&data[i].stream) );

		checkCudaErrors( cudaMalloc( &data[i].valueCur, sizeof(float)*4 ) );

    	// Allocate all memory from data structure to current device
    	checkCudaErrors( cudaMalloc( &data[i].ux, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].uy, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].rho, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].solid, sizeof(bool)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].solid_bound, sizeof(bool)*size ) );

    	// Multiphase fluid
    	checkCudaErrors( cudaMalloc( &data[i].psi, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].ueq_x, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].ueq_y, sizeof(float)*size ) );

    	if(i != 0) checkCudaErrors(
    			cudaMalloc(&data[i].plotBuffer, sizeof(unsigned int)*size*2) );

    	checkCudaErrors( cudaMalloc( &data[i].f0, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].f1, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].f2, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].f3, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].f4, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].f5, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].f6, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].f7, sizeof(float)*size ) );
    	checkCudaErrors( cudaMalloc( &data[i].f8, sizeof(float)*size ) );

    	checkCudaErrors( cudaMalloc( &data[i].border_f1, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f5, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f8, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f3, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f6, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f7, sizeof(float)*height ) );

    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f1, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f5, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f8, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f3, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f6, sizeof(float)*height ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f7, sizeof(float)*height ) );

    	// Copy initial data from host to the device
    	checkCudaErrors( cudaMemcpy( data[i].ux, ux , sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].uy, uy , sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].rho, rho , sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].solid, solid, sizeof(bool)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].solid_bound, solid_bound, sizeof(bool)*size, cudaMemcpyDefault ) );

    	// Multiphase fluid
    	checkCudaErrors( cudaMemcpy( data[i].psi, psi, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].ueq_x, ueq_x, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].ueq_y, ueq_y, sizeof(float)*size, cudaMemcpyDefault ) );

    	checkCudaErrors( cudaMemcpy( data[i].f0, f0, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].f1, f1, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].f2, f2, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].f3, f3, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].f4, f4, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].f5, f5, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].f6, f6, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].f7, f7, sizeof(float)*size, cudaMemcpyDefault ) );
    	checkCudaErrors( cudaMemcpy( data[i].f8, f8, sizeof(float)*size, cudaMemcpyDefault ) );
    }

	// Deallocate host memories
    delete [] ux;
    delete [] uy;
    delete [] rho;
    delete [] solid;
    delete [] solid_bound;
    delete [] f0;
    delete [] f1;
    delete [] f2;
    delete [] f3;
    delete [] f4;
    delete [] f5;
    delete [] f6;
    delete [] f7;
    delete [] f8;
    // Multiphase fluid
    delete [] psi;
    delete [] ueq_x;
    delete [] ueq_y;
}

void SimulationD2Q9::runLBM()
{
	int left, right;

	int subStep, dev;

	float ux_in = input->getUx();
	float uy_in = input->getUy();
	float rho_out = input->getRho();
	float tauInv = input->getTauInv();
	int updateSteps = input->getUpdateSteps();

	// External forces
	float Fx = input->getExtFx();
	float Fy = input->getExtFy();

	// Multiphase fluid
	float G = input->getG();
	float Gads = input->getGads();

	float minU = input->getMinU();
	float maxU = input->getMaxU();

	float minRho = input->getMinRho();
	float maxRho = input->getMaxRho();

	int xCur, yCur, zCur;
	float valueCur[4];

	output->setUpdateSteps(updateSteps);

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGraphicsMapResources(deviceCount, cuda_pbo_resource, 0));
	for(dev=0; dev < deviceCount; dev++)
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
				(void **)&plotData[dev], &num_bytes, cuda_pbo_resource[dev]));

	output->initLBMTime();
	// Run 'subSteps' times without show
	for(subStep=0; subStep < updateSteps; subStep++)
	{
		// Run streaming process for each device
		for(dev=0; dev < deviceCount; dev++)
		{
			//checkCudaErrors(cudaSetDevice(dev));
			streaming_d2q9(&data[dev], grid, block);
		}

		//Exchange borders
		if (deviceCount > 1 || commSize > 1)
		{
			// Get the border for each device
			for(dev=0; dev < deviceCount; dev++)
			{
				//checkCudaErrors(cudaSetDevice(dev));
				get_border_d2q9(&data[dev], grid, block);
			}

			// Copy border data between neighbor devices
			for(dev=0; dev < deviceCount; dev++)
			{
				if (deviceCount == 1) { left = 0; right = 0; }
				else
				{
					if (dev > 0 && dev < (deviceCount-1)) { left = dev-1; right = dev+1; }
					if (dev == 0) { left = deviceCount-1; right = dev+1; }
					if (dev == deviceCount-1) { left = dev-1; right = 0; }
				}

				//checkCudaErrors(cudaSetDevice(dev));
				copy_border_devices_d2q9(
						data[dev].nb_border_f1, data[dev].nb_border_f5,
						data[dev].nb_border_f8,	data[dev].nb_border_f3,
						data[dev].nb_border_f6, data[dev].nb_border_f7,
						data[right].border_f1, data[right].border_f5,
						data[right].border_f8, data[left].border_f3,
						data[left].border_f6, data[left].border_f7,
						height, data[dev].stream);
			}

			checkCudaErrors( cudaDeviceSynchronize() );
			/*if (commSize > 1)
			{
				// Copy border data between neighbor nodes
				if (commRank == 0)    { left = commSize-1; right = commRank+1; }
				if (commRank == (commSize-1)) { left = commRank-1; right = 0; }
				if (commRank > 0 && commRank < (commSize-1)) { left = commRank-1; right = commRank+1; }
				last = deviceCount-1;

				checkCudaErrors( cudaSetDevice( 0 ) );
				copy_border_nodes( border_f1_buf, border_f5_buf, border_f8_buf,
						data[0].nb_border_f1, data[0].nb_border_f5, data[0].nb_border_f8,
						height, right, left, Stat, data[0].stream);

				checkCudaErrors( cudaSetDevice( last ) );
				copy_border_nodes( border_f3_buf, border_f6_buf, border_f7_buf,
						data[last].nb_border_f3, data[last].nb_border_f6, data[last].nb_border_f7,
						height, left, right, Stat, data[last].stream);
			}*/

			// Apply the border received from neighbor devices
			for(dev=0; dev < deviceCount; dev++)
			{
				//checkCudaErrors(cudaSetDevice(dev));
				apply_border_d2q9(&data[dev], grid, block);
			}

		}
		checkCudaErrors( cudaDeviceSynchronize() );

		// Apply bounce-back, calculate the velocities and the equilibrium
		// distribution function and run collision process
		for(dev=0; dev < deviceCount; dev++)
		{
			//checkCudaErrors(cudaSetDevice(dev));
			if(dev == 0) collision_d2q9(plotData[dev], &data[dev], w0, w1, w2, tauInv, ux_in, uy_in, rho_out,
					minU, maxU, minRho, maxRho, grid, block, Fx, Fy, G, Gads);
			else
			{
				collision_d2q9(data[dev].plotBuffer, &data[dev], w0, w1, w2, tauInv, ux_in, uy_in, rho_out,
						minU, maxU, minRho, maxRho, grid, block, Fx, Fy, G, Gads);
				checkCudaErrors( cudaMemcpy( plotData[dev], data[dev].plotBuffer, sizeof(unsigned int)*size*2, cudaMemcpyDefault ) );
			}
		}

		checkCudaErrors(cudaDeviceSynchronize());
		cursor->getPos(&xCur, &yCur, &zCur);
		read_value_2d(plotData[0], &data[0], xCur, yCur, zCur);
		checkCudaErrors( cudaMemcpy( &valueCur, data[0].valueCur, sizeof(float)*4, cudaMemcpyDefault ) );
		input->setCursorValue(valueCur);
		steps++;
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGraphicsUnmapResources(deviceCount, cuda_pbo_resource, 0));

	output->setLBMTime();
}


void SimulationD2Q9::saveVtk()
{
	float *dataUx = new float[size];
	float *dataUy = new float[size];
	float *dataRho = new float[size];

	output->saveParVTK();
	for(int dev=0; dev < deviceCount; dev++) {
		checkCudaErrors(cudaMemcpy(dataUx, data[dev].ux, size*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(dataUy, data[dev].uy, size*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(dataRho, data[dev].rho, size*sizeof(float), cudaMemcpyDefault));
		output->setData(dataUx, dataUy, NULL, dataRho);
		output->saveVTK(dev);
	}
	delete dataUx;
	delete dataUy;
	delete dataRho;
}


