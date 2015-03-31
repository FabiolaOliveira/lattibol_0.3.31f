/***************************************************************************
 *   Copyright (C) 2013 by Lucas Monteiro Volpe             		       *
 *   lucasmvolpe@gmail.com                                                 *
 *   UNICAMP - Universidade Estadual de Campinas			   			   *
 *                       						   						   *
 *   This file is part of lattibol.					   					   *
 *                                                  			   		   *
 *   lattibol is free software: you can redistribute it and/or modify	   *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation, either version 3 of the License, or     *
 *   (at your option) any later version.				   				   *
 *									   									   *
 *   lattibol is distributed in the hope that it will be useful,	       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of	       *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	       *
 *   GNU General Public License for more details.			    		   *
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

#include "simulation_d3q19.h"

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

SimulationD3Q19::SimulationD3Q19(Input* inputPtr) : Simulation(inputPtr)
{
	data = new DataD3Q19[deviceCount];

	w0 = 12./36.;
	w1 = 2./36.;
	w2 = 1./36.;

	int i;
	int xn, x, xp, yn, y, yp, zn, z, zp;

	// Create variables on host side
	float *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8, *f9;
	float *f10, *f11, *f12, *f13, *f14, *f15, *f16, *f17, *f18;
	float *ux, *uy, *uz, *rho;
	bool *solid;
	bool *solid_bound;

	// Allocate memories on host side
	ux = new float[size];
	uy = new float[size];
	uz = new float[size];
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
	f9 = new float[size];
	f10 = new float[size];
	f11 = new float[size];
	f12 = new float[size];
	f13 = new float[size];
	f14 = new float[size];
	f15 = new float[size];
	f16 = new float[size];
	f17 = new float[size];
	f18 = new float[size];
	solid = new bool[size];

	for(i=0; i < size; i++) solid_bound[i]=0;

	bool *allSolids = input->getSolids();

	ux[0] = input->getUx();
	uy[0] = input->getUy();
	uz[0] = input->getUz();
	rho[0] = input->getRho();

	f0[0] = w0*rho[0]*( 1.                                                - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f1[0] = w1*rho[0]*( 1. + 3.*(+ux[0]      ) + 4.5*(+ux[0]      )*(+ux[0]      ) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f2[0] = w1*rho[0]*( 1. + 3.*(-ux[0]      ) + 4.5*(-ux[0]      )*(-ux[0]      ) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f3[0] = w1*rho[0]*( 1. + 3.*(   +uy[0]   ) + 4.5*(   +uy[0]   )*(   +uy[0]   ) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f4[0] = w1*rho[0]*( 1. + 3.*(   -uy[0]   ) + 4.5*(   -uy[0]   )*(   -uy[0]   ) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f5[0] = w1*rho[0]*( 1. + 3.*(      +uz[0]) + 4.5*(      +uz[0])*(      +uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f6[0] = w1*rho[0]*( 1. + 3.*(      -uz[0]) + 4.5*(      -uz[0])*(      -uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f7[0] = w2*rho[0]*( 1. + 3.*(+ux[0]+uy[0]   ) + 4.5*(+ux[0]+uy[0]   )*(+ux[0]+uy[0]   ) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f8[0] = w2*rho[0]*( 1. + 3.*(+ux[0]-uy[0]   ) + 4.5*(+ux[0]-uy[0]   )*(+ux[0]-uy[0]   ) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f9[0] = w2*rho[0]*( 1. + 3.*(+ux[0]   +uz[0]) + 4.5*(+ux[0]   +uz[0])*(+ux[0]   +uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f10[0]= w2*rho[0]*( 1. + 3.*(+ux[0]   -uz[0]) + 4.5*(+ux[0]   -uz[0])*(+ux[0]   -uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f11[0]= w2*rho[0]*( 1. + 3.*(-ux[0]+uy[0]   ) + 4.5*(-ux[0]+uy[0]   )*(-ux[0]+uy[0]   ) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f12[0]= w2*rho[0]*( 1. + 3.*(-ux[0]-uy[0]   ) + 4.5*(-ux[0]-uy[0]   )*(-ux[0]-uy[0]   ) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f13[0]= w2*rho[0]*( 1. + 3.*(-ux[0]   +uz[0]) + 4.5*(-ux[0]   +uz[0])*(-ux[0]   +uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f14[0]= w2*rho[0]*( 1. + 3.*(-ux[0]   -uz[0]) + 4.5*(-ux[0]   -uz[0])*(-ux[0]   -uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f15[0]= w2*rho[0]*( 1. + 3.*(   +uy[0]+uz[0]) + 4.5*(   +uy[0]+uz[0])*(   +uy[0]+uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f16[0]= w2*rho[0]*( 1. + 3.*(   +uy[0]-uz[0]) + 4.5*(   +uy[0]-uz[0])*(   +uy[0]-uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f17[0]= w2*rho[0]*( 1. + 3.*(   -uy[0]+uz[0]) + 4.5*(   -uy[0]+uz[0])*(   -uy[0]+uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );
	f18[0]= w2*rho[0]*( 1. + 3.*(   -uy[0]-uz[0]) + 4.5*(   -uy[0]-uz[0])*(   -uy[0]-uz[0]) - 1.5*(ux[0]*ux[0]+uy[0]*uy[0]+uz[0]*uz[0]) );

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
		f9[i] = f9[0];
		f10[i] = f10[0];
		f11[i] = f11[0];
		f12[i] = f12[0];
		f13[i] = f13[0];
		f14[i] = f14[0];
		f15[i] = f15[0];
		f16[i] = f16[0];
		f17[i] = f17[0];
		f18[i] = f18[0];
	}

	for (i = 0; i < deviceCount; i++)
	{
    	//checkCudaErrors(cudaSetDevice(i));

		data[i].width  = width;
		data[i].height = height;
		data[i].depth = depth;
		data[i].deviceCount = deviceCount;
		data[i].dev = i;

		// Get subdomain solids
		for (z=0; z < depth; z++) for (y=0; y<height; y++) for (x=0; x<width; x++)
			solid[x+y*width+z*width*height] =
					allSolids[(x + i*width)+y*width*deviceCount+z*width*deviceCount*height];

		// If all boundaries are solid, solid_bound=1;
		for (z=0; z < depth; z++) for (y=0; y<height; y++) for (x=1; x<width-1; x++)
		{
			xn = x-1;
			xp = x+1;
			// If y is the most top node, the top neighbor will be the most bottom node
			yn = (y > 0         ) ? (y-1) : (height-1);
			// If y is the most bottom node, the bottom neighbor will be the most top node
			yp = (y < (height-1)) ? (y+1) : (0       );
			// If y is the most top node, the top neighbor will be the most bottom node
			zn = (z > 0         ) ? (z-1) : (depth-1);
			// If y is the most bottom node, the bottom neighbor will be the most top node
			zp = (z < (depth-1 )) ? (z+1) : (0       );

			if
			(
					solid[ x  + y *width + z *width*height ] &&
					solid[ xp + y *width + z *width*height ] &&
					solid[ xn + y *width + z *width*height ] &&
					solid[ x  + yp*width + z *width*height ] &&
					solid[ x  + yn*width + z *width*height ] &&
					solid[ x  + y *width + zp*width*height ] &&
					solid[ x  + y *width + zn*width*height ] &&
					solid[ xp + yp*width + z *width*height ] &&
					solid[ xp + yn*width + z *width*height ] &&
					solid[ xp + y *width + zp*width*height ] &&
					solid[ xp + y *width + zn*width*height ] &&
					solid[ xn + yp*width + z *width*height ] &&
					solid[ xn + yn*width + z *width*height ] &&
					solid[ xn + y *width + zp*width*height ] &&
					solid[ xn + y *width + zn*width*height ] &&
					solid[ x  + yp*width + zp*width*height ] &&
					solid[ x  + yp*width + zn*width*height ] &&
					solid[ x  + yn*width + zp*width*height ] &&
					solid[ x  + yn*width + zn*width*height ]
			)
				solid_bound[x+y*width+z*width*height] = 1;
			else solid_bound[x+y*width+z*width*height] = 0;
		}

        checkCudaErrors( cudaStreamCreate(&data[i].stream) );

		// Allocate all memory from data structure to current device
		checkCudaErrors( cudaMalloc( &data[i].valueCur, sizeof(float)*4 ) );

		checkCudaErrors( cudaMalloc( &data[i].ux, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].uy, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].uz, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].rho, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].solid, sizeof(bool)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].solid_bound, sizeof(bool)*size ) );

		if(i != 0) checkCudaErrors(
				cudaMalloc(&data[i].plotBuffer, sizeof(unsigned int)*size) );

		checkCudaErrors( cudaMalloc( &data[i].f0, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f1, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f2, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f3, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f4, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f5, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f6, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f7, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f8, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f9, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f10, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f11, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f12, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f13, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f14, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f15, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f16, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f17, sizeof(float)*size ) );
		checkCudaErrors( cudaMalloc( &data[i].f18, sizeof(float)*size ) );

    	checkCudaErrors( cudaMalloc( &data[i].border_f1,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f7,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f8,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f9,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f10, sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f2,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f11, sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f12, sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f13, sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].border_f14, sizeof(float)*height*depth ) );

    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f1,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f7,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f8,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f9,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f10, sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f2,  sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f11, sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f12, sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f13, sizeof(float)*height*depth ) );
    	checkCudaErrors( cudaMalloc( &data[i].nb_border_f14, sizeof(float)*height*depth ) );

		// Copy initial data from host to the device
		checkCudaErrors( cudaMemcpy( data[i].ux, ux , sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].uy, uy , sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].uz, uz , sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].rho, rho , sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].solid, solid, sizeof(bool)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].solid_bound, solid_bound, sizeof(bool)*size, cudaMemcpyDefault ) );

		checkCudaErrors( cudaMemcpy( data[i].f0, f0, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f1, f1, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f2, f2, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f3, f3, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f4, f4, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f5, f5, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f6, f6, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f7, f7, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f8, f8, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f9, f9, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f10, f10, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f11, f11, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f12, f12, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f13, f13, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f14, f14, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f15, f15, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f16, f16, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f17, f17, sizeof(float)*size, cudaMemcpyDefault ) );
		checkCudaErrors( cudaMemcpy( data[i].f18, f18, sizeof(float)*size, cudaMemcpyDefault ) );
	}

	// Deallocate host memories
	delete [] ux;
	delete [] uy;
	delete [] uz;
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
	delete [] f9;
	delete [] f10;
	delete [] f11;
	delete [] f12;
	delete [] f13;
	delete [] f14;
	delete [] f15;
	delete [] f16;
	delete [] f17;
	delete [] f18;

}

void SimulationD3Q19::runLBM()
{
	int left, right;
	int subStep, dev;

	float ux_in = input->getUx();
	float uy_in = input->getUy();
	float uz_in = input->getUz();
	float rho_out = input->getRho();
	float tauInv = input->getTauInv();
	int updateSteps = input->getUpdateSteps();

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
	// Run 'step' times without show
	for(subStep=0; subStep < updateSteps; subStep++)
	{
		for(dev=0; dev < deviceCount; dev++)
			// Run streaming process for each device
			for(dev=0; dev < deviceCount; dev++)
			{
				//checkCudaErrors(cudaSetDevice(dev));
				streaming_d3q19(&data[dev], grid, block);
			}

		//Exchange borders
		if (deviceCount > 1)
		{
			// Get the border for each device
			for(dev=0; dev < deviceCount; dev++)
			{
				//checkCudaErrors(cudaSetDevice(dev));
				get_border_d3q19(&data[dev], grid, block);
			}
			checkCudaErrors( cudaDeviceSynchronize() );

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
				copy_border_devices_d3q19(
						data[dev].nb_border_f1, data[dev].nb_border_f7, data[dev].nb_border_f8,
						data[dev].nb_border_f9, data[dev].nb_border_f10,
						data[dev].nb_border_f2, data[dev].nb_border_f11, data[dev].nb_border_f12,
						data[dev].nb_border_f13, data[dev].nb_border_f14,
						data[right].border_f1, data[right].border_f7, data[right].border_f8,
						data[right].border_f9, data[right].border_f10,
						data[left].border_f2, data[left].border_f11, data[left].border_f12,
						data[left].border_f13, data[left].border_f14,
						height, depth, data[dev].stream);
			}

			checkCudaErrors( cudaDeviceSynchronize() );

			// Apply the border received from neighbor devices
			for(dev=0; dev < deviceCount; dev++)
			{
				//checkCudaErrors(cudaSetDevice(dev));
				apply_border_d3q19(&data[dev], grid, block);
			}

		}
		checkCudaErrors( cudaDeviceSynchronize() );

		// Apply bounce-back, calculate the velocities and the equilibrium
		// distribution function and run collision process
		for(dev=0; dev < deviceCount; dev++)
		{
			//checkCudaErrors(cudaSetDevice(dev));
			if(dev == 0) collision_d3q19(plotData[dev], &data[dev], w0, w1, w2, tauInv, ux_in, uy_in, uz_in, rho_out,
					minU, maxU, minRho, maxRho, grid, block);
			else
			{
				collision_d3q19(data[dev].plotBuffer, &data[dev], w0, w1, w2, tauInv, ux_in, uy_in, uz_in, rho_out,
						minU, maxU, minRho, maxRho, grid, block);
				checkCudaErrors( cudaMemcpy( plotData[dev], data[dev].plotBuffer, sizeof(unsigned int)*size, cudaMemcpyDefault ) );
			}
		}

		checkCudaErrors(cudaDeviceSynchronize());
		cursor->getPos(&xCur, &yCur, &zCur);
		read_value_3d(plotData[0], &data[0], xCur, yCur, zCur);
		checkCudaErrors( cudaMemcpy( &valueCur, data[0].valueCur, sizeof(float)*4, cudaMemcpyDefault ) );
		input->setCursorValue(valueCur);
		steps++;
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGraphicsUnmapResources(deviceCount, cuda_pbo_resource, 0));

	output->setLBMTime();
}

void SimulationD3Q19::saveVtk()
{
	float *dataUx = new float[size];
	float *dataUy = new float[size];
	float *dataUz = new float[size];
	float *dataRho = new float[size];

	output->saveParVTK();
	for(int dev=0; dev < deviceCount; dev++) {
		checkCudaErrors(cudaMemcpy(dataUx, data[dev].ux, size*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(dataUy, data[dev].uy, size*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(dataUz, data[dev].uz, size*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(dataRho, data[dev].rho, size*sizeof(float), cudaMemcpyDefault));
		output->setData(dataUx, dataUy, dataUz, dataRho);
		output->saveVTK(dev);
	}
	delete dataUx;
	delete dataUy;
	delete dataUz;
	delete dataRho;
}


