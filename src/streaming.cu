/***************************************************************************
 *   Copyright (C) 2014 by Fabíola Martins Campos de Oliveira and Lucas    *
 *   Monteiro Volpe              					 					   *
 *   fabiola.bass@gmail.com, lucasmvolpe@gmail.com                         *
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
 *   GNU General Public License for more details.			               *
 *									    								   *
 *   You should have received a copy of the GNU General Public License     *
 *   long with lattibol.  If not, see <http://www.gnu.org/licenses/>.      *
 ***************************************************************************/

#include "lattibol.h"

__global__ void streaming_d2q9_kernel ( DataD2Q9 data )
{
	// Create variables that defines neighbors and current coordinates
	int xn, x, xp, y, yp;
	int offset[5];
	int dim;
	
	x = threadIdx.x + blockIdx.x * blockDim.x;
	y = threadIdx.y + blockIdx.y * blockDim.y;

	dim = blockDim.x * gridDim.x;

	offset[0] = (x ) + (y ) * dim;

	if ( !data.solid_bound[offset[0]] )
	{

		float temp;

		// If x is the most left node, the left neighbor will be the most right node
		xn = (x > 0              ) ? (x-1) : (data.width-1 );
		// If x is the most right node, the right neighbor will be the most left node
		xp = (x < (data.width-1) ) ? (x+1) : (0       );
		// If y is the most bottom node, the bottom neighbor will be the most top node
		yp = (y > 0              ) ? (y-1) : (data.height-1 );

		// Calculate offset for fx_new
		offset[1] = (xp) + (y ) * dim;
		offset[2] = (x ) + (yp) * dim;
		offset[3] = (xp) + (yp) * dim;
		offset[4] = (xn) + (yp) * dim;

		// Apply streaming process moving current densities functions to next position
		temp = data.f1[offset[0]];
		data.f1[offset[0]] = data.f3[offset[1]];
		data.f3[offset[1]] = temp;
		temp = data.f2[offset[0]];
		data.f2[offset[0]] = data.f4[offset[2]];
		data.f4[offset[2]] = temp;
		temp = data.f5[offset[0]];
		data.f5[offset[0]] = data.f7[offset[3]];
		data.f7[offset[3]] = temp;
		temp = data.f6[offset[0]];
		data.f6[offset[0]] = data.f8[offset[4]];
		data.f8[offset[4]] = temp;

		// Multiphase fluid
		/*if (G != 0)
    		data.psi[offset[0]] = 4. * exp(- 200. / data.rho[offset[0]]);*/
	}
}


// C streaming function that calls CUDA kernel
extern "C"
void streaming_d2q9( DataD2Q9 *data, dim3 grid, dim3 block )
{
    streaming_d2q9_kernel<<<grid, block>>> ( *data );
}

__global__ void streaming_d3q19_kernel ( DataD3Q19 data )
{
	// Create variables that defines neighbors and current coordinates
	int x, xp, yn, y, yp, zn, z, zp;
	int offset[10];
	float temp;

	x = threadIdx.x + blockIdx.x * blockDim.x;
	y = threadIdx.y + blockIdx.y * blockDim.y;
	z = threadIdx.z + blockIdx.z * blockDim.z;

	int x_dim  = blockDim.x * gridDim.x;
	int xy_dim = x_dim*blockDim.y * gridDim.y;

	offset[0] = (x ) + (y )*x_dim + (z )*xy_dim;

	if ( !data.solid_bound[offset[0]] )
	{
		// Apply periodic boundary condition

		// If x is the most right node, the right neighbor will be the most left node
		xp = (x < (data.width-1) ) ? (x+1) : (0       );
		// If y is the most top node, the top neighbor will be the most bottom node
		yn = (y > 0         ) ? (y-1) : (data.height-1);
		// If y is the most bottom node, the bottom neighbor will be the most top node
		yp = (y < (data.height-1)) ? (y+1) : (0       );
		// If y is the most top node, the top neighbor will be the most bottom node
		zn = (z > 0         ) ? (z-1) : (data.depth-1);
		// If y is the most bottom node, the bottom neighbor will be the most top node
		zp = (z < (data.depth-1)) ? (z+1) : (0       );

		// Calculate offset for fx_new
		offset[1] = (xp) + (y )*x_dim + (z )*xy_dim;
		offset[2] = (x ) + (yp)*x_dim + (z )*xy_dim;
		offset[3] = (x ) + (y )*x_dim + (zp)*xy_dim;
		offset[4] = (xp) + (yp)*x_dim + (z )*xy_dim;
		offset[5] = (xp) + (yn)*x_dim + (z )*xy_dim;
		offset[6] = (xp) + (y )*x_dim + (zp)*xy_dim;
		offset[7] = (xp) + (y )*x_dim + (zn)*xy_dim;
		offset[8] = (x ) + (yp)*x_dim + (zp)*xy_dim;
		offset[9] = (x ) + (yp)*x_dim + (zn)*xy_dim;

		// Apply streaming process moving current densities functions to next position
		temp = data.f1[offset[0]];
		data.f1[offset[0]] = data.f2[offset[1]];
		data.f2[offset[1]] = temp;

		temp = data.f3[offset[0]];
		data.f3[offset[0]] = data.f4[offset[2]];
		data.f4[offset[2]] = temp;

		temp = data.f5[offset[0]];
		data.f5[offset[0]] = data.f6[offset[3]];
		data.f6[offset[3]] = temp;

		temp = data.f7[offset[0]];
		data.f7[offset[0]] = data.f12[offset[4]];
		data.f12[offset[4]] = temp;

		temp = data.f8[offset[0]];
		data.f8[offset[0]] = data.f11[offset[5]];
		data.f11[offset[5]] = temp;

		temp = data.f9[offset[0]];
		data.f9[offset[0]] = data.f14[offset[6]];
		data.f14[offset[6]] = temp;

		temp = data.f10[offset[0]];
		data.f10[offset[0]] = data.f13[offset[7]];
		data.f13[offset[7]] = temp;

		temp = data.f15[offset[0]];
		data.f15[offset[0]] = data.f18[offset[8]];
		data.f18[offset[8]] = temp;

		temp = data.f16[offset[0]];
		data.f16[offset[0]] = data.f17[offset[9]];
		data.f17[offset[9]] = temp;
	}
}

extern "C"
void streaming_d3q19( DataD3Q19 *data, dim3 grid, dim3 block )
{
    streaming_d3q19_kernel<<<grid, block>>> ( *data );
}
