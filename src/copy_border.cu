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
#include <mpi.h>

extern "C"
void copy_border_devices_d2q9(
		float *nb_border_f1, float *nb_border_f5, float *nb_border_f8,
		float *nb_border_f3, float *nb_border_f6, float *nb_border_f7,
		float *border_f1, float *border_f5, float *border_f8,
		float *border_f3, float *border_f6, float *border_f7,
		int height, cudaStream_t &s )
{
	// Copies border_fx to nb_border_fx using UVA
	checkCudaErrors(cudaMemcpyAsync(nb_border_f1, border_f1, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f5, border_f5, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f8, border_f8, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f3, border_f3, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f6, border_f6, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f7, border_f7, sizeof(float)*height,
			cudaMemcpyDefault, s));
}

void copy_border_devices_d3q19(
		float *nb_border_f1, float *nb_border_f7, float *nb_border_f8,
		float *nb_border_f9, float *nb_border_f10,
		float *nb_border_f2, float *nb_border_f11, float *nb_border_f12,
		float *nb_border_f13, float *nb_border_f14,
		float *border_f1, float *border_f7, float *border_f8,
		float *border_f9, float *border_f10,
		float *border_f2, float *border_f11, float *border_f12,
		float *border_f13, float *border_f14,
		int height, int depth, cudaStream_t &s )
{
	// Copies border_fx to nb_border_fx using UVA
	checkCudaErrors(cudaMemcpyAsync(nb_border_f1, border_f1, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f7, border_f7, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f8, border_f8, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f9, border_f9, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f10, border_f10, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f2, border_f2, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f11, border_f11, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f12, border_f12, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f13, border_f13, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_f14, border_f14, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
}

void copy_border_nodes_d2q9(
		float *border_ctr_buf, float *border_top_buf, float *border_btm_buf,
		float *nb_border_ctr, float *nb_border_top, float *nb_border_btm,
		int height, int dest, int source, cudaStream_t &s)
{
	checkCudaErrors(cudaMemcpyAsync(border_ctr_buf, nb_border_ctr, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(border_top_buf, nb_border_top, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(border_btm_buf, nb_border_btm, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors( cudaDeviceSynchronize() );

	MPI_Sendrecv_replace(border_ctr_buf, height, MPI_FLOAT, dest, 1, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Sendrecv_replace(border_top_buf, height, MPI_FLOAT, dest, 1, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Sendrecv_replace(border_btm_buf, height, MPI_FLOAT, dest, 1, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	checkCudaErrors(cudaMemcpyAsync(nb_border_ctr, border_ctr_buf, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_top, border_top_buf, sizeof(float)*height,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_btm, border_btm_buf, sizeof(float)*height,
			cudaMemcpyDefault, s));
}

void copy_border_nodes_d3q19(
		float *border_ctr_buf, float *border_lft_buf, float *border_rgt_buf,
		float *border_top_buf, float *border_btm_buf,
		float *nb_border_ctr, float *nb_border_lft, float *nb_border_rgt,
		float *nb_border_top, float *nb_border_btm,
		int height, int depth, int dest, int source, cudaStream_t &s)
{
	checkCudaErrors(cudaMemcpyAsync(border_ctr_buf, nb_border_ctr, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(border_lft_buf, nb_border_lft, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(border_rgt_buf, nb_border_rgt, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(border_top_buf, nb_border_top, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(border_btm_buf, nb_border_btm, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));

	checkCudaErrors( cudaDeviceSynchronize() );

	MPI_Sendrecv_replace(border_ctr_buf, height*depth, MPI_FLOAT, dest, 1, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Sendrecv_replace(border_lft_buf, height*depth, MPI_FLOAT, dest, 1, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Sendrecv_replace(border_rgt_buf, height*depth, MPI_FLOAT, dest, 1, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Sendrecv_replace(border_top_buf, height*depth, MPI_FLOAT, dest, 1, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Sendrecv_replace(border_btm_buf, height*depth, MPI_FLOAT, dest, 1, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	checkCudaErrors(cudaMemcpyAsync(nb_border_ctr, border_ctr_buf, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_lft, border_lft_buf, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_rgt, border_rgt_buf, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_top, border_top_buf, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
	checkCudaErrors(cudaMemcpyAsync(nb_border_btm, border_btm_buf, sizeof(float)*height*depth,
			cudaMemcpyDefault, s));
}

