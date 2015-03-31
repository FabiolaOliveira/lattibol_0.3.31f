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

#include "read_value.h"

#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_cuda_gl.h>

__global__ void read_value_2d_kernel (float *plot_data, DataD2Q9 data, int xPos, int yPos, int zPos)
{
	int offset = xPos + yPos*data.width+ zPos*data.width*data.height;
	data.valueCur[0] = plot_data[offset*4 ];
	data.valueCur[1] = plot_data[offset*4+1];
	data.valueCur[2] = plot_data[offset*4+2];
	data.valueCur[3] = plot_data[offset*4+3];
}

extern "C"
void read_value_2d(float *plot_data, DataD2Q9 *data, int x, int y, int z)
{
	read_value_2d_kernel<<<1, 1>>> (plot_data, *data, x, y, z);
	checkCudaErrors(cudaDeviceSynchronize());
}


__global__ void read_value_3d_kernel (float *plot_data, DataD3Q19 data, int xPos, int yPos, int zPos)
{
	int offset = xPos + yPos*data.width+ zPos*data.width*data.height;
	data.valueCur[0] = plot_data[offset*4 ];
	data.valueCur[1] = plot_data[offset*4+1];
	data.valueCur[2] = plot_data[offset*4+2];
	data.valueCur[3] = plot_data[offset*4+3];
}

extern "C"
void read_value_3d(float *plot_data, DataD3Q19 *data, int x, int y, int z)
{
	read_value_3d_kernel<<<1, 1>>> (plot_data, *data, x, y, z);
	checkCudaErrors(cudaDeviceSynchronize());
}

