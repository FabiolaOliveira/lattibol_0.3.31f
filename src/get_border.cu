/***************************************************************************
 *   Copyright (C) 2013 by Fabíola Martins Campos de Oliveira and Lucas    *
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

#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "lattibol.h"

__global__ void get_border_d2q9_kernel ( DataD2Q9 data )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	// if x is the most right node copies (f1, f5  and f8) fx_new to border_fx 
	if (x == (data.width-1))
	{
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;

		data.border_f1[y] = data.f1[offset];
		data.border_f5[y] = data.f5[offset];
		data.border_f8[y] = data.f8[offset];
	}
	// if x is the most left node copies (f3, f6  and f7) fx_new to border_fx 
	if (x == 0)
	{
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;

		data.border_f3[y] = data.f3[offset];
		data.border_f6[y] = data.f6[offset];
		data.border_f7[y] = data.f7[offset];
	}
}

// C get_border function that calls CUDA kernel
extern "C" 
void get_border_d2q9( DataD2Q9 *data, dim3 grid, dim3 block )
{
	get_border_d2q9_kernel<<<grid, block>>> ( *data );
}

__global__ void get_border_d3q19_kernel ( DataD3Q19 data )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	// if x is the most right node copies (f1, f5  and f8) fx_new to border_fx
	if (x == (data.width-1))
	{
		int y = threadIdx.y + blockIdx.y*blockDim.y;
		int z = threadIdx.z + blockIdx.z*blockDim.z;
		int width = blockDim.x*gridDim.x;
		int height = blockDim.y*gridDim.y;

		int offset = x + y*width + z*width*height;
		int bd_offset = y + z*height;

		data.border_f1 [bd_offset] = data.f1 [offset];
		data.border_f7 [bd_offset] = data.f7 [offset];
		data.border_f8 [bd_offset] = data.f8 [offset];
		data.border_f9 [bd_offset] = data.f9 [offset];
		data.border_f10[bd_offset] = data.f10[offset];

	}
	// if x is the most left node copies (f3, f6  and f7) fx to border_fx
	if (x == 0)
	{
		int y = threadIdx.y + blockIdx.y*blockDim.y;
		int z = threadIdx.z + blockIdx.z*blockDim.z;
		int width = blockDim.x*gridDim.x;
		int height = blockDim.y*gridDim.y;

		int offset = x + y*width + z*width*height;
		int bd_offset = y + z*height;

		data.border_f2 [bd_offset] = data.f2 [offset];
		data.border_f11[bd_offset] = data.f11[offset];
		data.border_f12[bd_offset] = data.f12[offset];
		data.border_f13[bd_offset] = data.f13[offset];
		data.border_f14[bd_offset] = data.f14[offset];
	}
}

extern "C"
void get_border_d3q19( DataD3Q19 *data, dim3 grid, dim3 block )
{
	get_border_d3q19_kernel<<<grid, block>>> ( *data );
}
