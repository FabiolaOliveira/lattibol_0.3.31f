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

#include "collision.h"

#include <stdio.h>

__device__
void float2RGBA(float *rgba, float ux, float uy, float uz, float rho )
{
	rgba[0] = ux;
	rgba[1] = uy;
	rgba[2] = uz;
	rgba[3] = rho;
}

__global__ void multiphase_collision_d2q9_kernel (float *plot_data, DataD2Q9 data,
		float w0, float w1,	float w5, float tau_inv, float ux_in, float uy_in,
		float rho_out, float u_min, float u_max, float rho_min, float rho_max,
		float Fx, float Fy, float G, float Gads)
{
	int width = blockDim.x * gridDim.x;
	int height = blockDim.y * gridDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y*width;

	int plot_offset = offset;
	int plot_offset2 = offset + width*height;

	if (!data.solid[offset])
	{
		float ueq_xueq_y5, ueq_xueq_y6;
		float ueq_x, ueq_y, ueq_sq, ueq_xsq, ueq_ysq;
		int xn, xp, yn, yp;
		int offsetM[9];
		float IFx = 0;
		float IFy = 0;

		// Periodic boundary
		// If x is the most left node, the left neighbor will be the most right node
		xn = (x > 0              ) ? (x-1) : (width-1 );
		// If x is the most right node, the right neighbor will be the most left node
		xp = (x < (width-1) ) ? (x+1) : (0       );
		// If y is the most top node, the top neighbor will be the most bottom node
		yn = (y > 0              ) ? (y-1) : (height-1 );
		// If y is the most bottom node, the bottom neighbor will be the most top node
		yp = (y < (height-1)) ? (y+1) : (0 );

		offsetM[0] = y  * width + x;
		offsetM[1] = y  * width + xp;
		offsetM[2] = yp * width + x;
		offsetM[3] = y  * width + xn;
		offsetM[4] = yn * width + x;
		offsetM[5] = yp * width + xp;
		offsetM[6] = yp * width + xn;
		offsetM[7] = yn * width + xn;
		offsetM[8] = yn * width + xp;
		
		IFx = IFx + w1 * data.psi[offsetM[1]];
		IFx = IFx - w1 * data.psi[offsetM[3]];
		IFx = IFx + w5 * data.psi[offsetM[5]];
		IFx = IFx - w5 * data.psi[offsetM[6]];
		IFx = IFx - w5 * data.psi[offsetM[7]];
		IFx = IFx + w5 * data.psi[offsetM[8]];
		
		IFy = IFy - w1 * data.psi[offsetM[2]];
		IFy = IFy + w1 * data.psi[offsetM[4]];
		IFy = IFy - w5 * data.psi[offsetM[5]];
		IFy = IFy - w5 * data.psi[offsetM[6]];
		IFy = IFy + w5 * data.psi[offsetM[7]];
		IFy = IFy + w5 * data.psi[offsetM[8]];

		IFx = - G * data.psi[offset] * IFx;
		IFy = - G * data.psi[offset] * IFy;

		ueq_x = data.ueq_x[offset] + (IFx / (tau_inv * data.rho[offset]));
		ueq_y = data.ueq_y[offset] + (IFy / (tau_inv * data.rho[offset]));
		// End interaction force

		if(Gads != 0)
		{
			// Fluid-Surface forces
			float Fsx = 0;
			float Fsy = 0;

			if (data.solid[offsetM[1]])
				Fsx = Fsx + w1;
			if (data.solid[offsetM[2]])
				Fsy = Fsy - w1;
			if (data.solid[offsetM[3]])
				Fsx = Fsx - w1;
			if (data.solid[offsetM[4]])
				Fsy = Fsy + w1;
			if (data.solid[offsetM[5]])
			{
				Fsx = Fsx + w5;
				Fsy = Fsy - w5;
			}
			if (data.solid[offsetM[6]])
			{
				Fsx = Fsx - w5;
				Fsy = Fsy - w5;
			}
			if (data.solid[offsetM[7]])
			{
				Fsx = Fsx - w5;
				Fsy = Fsy + w5;
			}
			if (data.solid[offsetM[8]])
			{
				Fsx = Fsx + w5;
				Fsy = Fsy + w5;
			}

			Fsx = - Gads * data.psi[offset] * Fsx;
			Fsy = - Gads * data.psi[offset] * Fsy;

			ueq_x = ueq_x + (Fsx / (tau_inv * data.rho[offset]));
			ueq_y = ueq_y + (Fsy / (tau_inv * data.rho[offset]));
		}

		// Calculate square velocity and macroscopic velocity
		ueq_xsq = ueq_x * ueq_x;
		ueq_ysq = ueq_y * ueq_y;
		ueq_sq  = ueq_xsq + ueq_ysq;

		float2RGBA(&plot_data[plot_offset*4], data.ux[offset], data.uy[offset], 0., data.rho[offset]);

		// Pre-calculate some values
		w0 = w0 * data.rho[offset];
		w1 = w1 * data.rho[offset];
		w5 = w5 * data.rho[offset];
		ueq_xueq_y5 = +ueq_x + ueq_y;
		ueq_xueq_y6 = -ueq_x + ueq_y;

		// Calculate the collision term and pass the new equilibrium distribution function
		data.f0[offset] = data.f0[offset] - tau_inv*( data.f0[offset] - w0*(1.f                                					 - 1.5f*ueq_sq) );
		data.f1[offset] = data.f1[offset] - tau_inv*( data.f1[offset] - w1*(1.f + 3.f*ueq_x    	  + 4.5f*ueq_xsq        		 - 1.5f*ueq_sq) );
		data.f2[offset] = data.f2[offset] - tau_inv*( data.f2[offset] -	w1*(1.f + 3.f*ueq_y    	  + 4.5f*ueq_ysq        		 - 1.5f*ueq_sq) );
		data.f3[offset] = data.f3[offset] - tau_inv*( data.f3[offset] -	w1*(1.f - 3.f*ueq_x       + 4.5f*ueq_xsq        		 - 1.5f*ueq_sq) );
		data.f4[offset] = data.f4[offset] - tau_inv*( data.f4[offset] -	w1*(1.f - 3.f*ueq_y       + 4.5f*ueq_ysq        		 - 1.5f*ueq_sq) );
		data.f5[offset] = data.f5[offset] - tau_inv*( data.f5[offset] -	w5*(1.f + 3.f*ueq_xueq_y5 + 4.5f*ueq_xueq_y5*ueq_xueq_y5 - 1.5f*ueq_sq) );
		data.f6[offset] = data.f6[offset] - tau_inv*( data.f6[offset] -	w5*(1.f + 3.f*ueq_xueq_y6 + 4.5f*ueq_xueq_y6*ueq_xueq_y6 - 1.5f*ueq_sq) );
		data.f7[offset] = data.f7[offset] - tau_inv*( data.f7[offset] -	w5*(1.f - 3.f*ueq_xueq_y5 + 4.5f*ueq_xueq_y5*ueq_xueq_y5 - 1.5f*ueq_sq) );
		data.f8[offset] = data.f8[offset] - tau_inv*( data.f8[offset] -	w5*(1.f - 3.f*ueq_xueq_y6 + 4.5f*ueq_xueq_y6*ueq_xueq_y6 - 1.5f*ueq_sq) );
	}
}

__global__ void collision_d2q9_kernel (float *plot_data, DataD2Q9 data,
		float w0, float w1,	float w5, float tau_inv, float ux_in, float uy_in,
		float rho_out, float u_min, float u_max, float rho_min, float rho_max,
		float Fx, float Fy, float G)
{
	int width = blockDim.x * gridDim.x;
	int height = blockDim.y * gridDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y*width;

	int plot_offset = offset;
	int plot_offset2 = offset + width*height;

	if (!data.solid[offset])
	{
		// Create registers
		float rho, ux, uy, ru;
		float temp;
		// External forces
		float ueq_xueq_y5, ueq_xueq_y6;
		float ueq_x, ueq_y, ueq_sq, ueq_xsq, ueq_ysq;
		// Multiphase fluid
		int xn, xp, yn, yp;

		//Apply Bounce-Back boundary condition
		temp = data.f1[offset];
		data.f1[offset] = data.f3[offset];
		data.f3[offset] = temp;

		temp = data.f2[offset];
		data.f2[offset] = data.f4[offset];
		data.f4[offset] = temp;

		temp = data.f5[offset];
		data.f5[offset] = data.f7[offset];
		data.f7[offset] = temp;

		temp = data.f6[offset];
		data.f6[offset] = data.f8[offset];
		data.f8[offset] = temp;

		// Velocity Boundary at west
		if (x == 0 && data.dev == 0)
		{
			rho = (data.f0[offset] + data.f2[offset] + data.f4[offset] +
					2.*(data.f3[offset] + data.f7[offset] + data.f6[offset])) /
							(1. - ux_in);

			ru = rho * ux_in;

			data.f1[offset] = data.f3[offset] + (2./3.)*ru;
			data.f5[offset] = data.f7[offset] + (1./6.)*ru - 0.5*(data.f2[offset] - data.f4[offset]);
			data.f8[offset] = data.f6[offset] + (1./6.)*ru - 0.5*(data.f4[offset] - data.f2[offset]);
		}

		// Pressure Boundary at west
		/*if (x == 0 && data.dev == 0)
		{
			ux = 1. - (data.f0[offset] + data.f2[offset] + data.f4[offset] +
					2.*(data.f3[offset] + data.f7[offset] + data.f6[offset])) / rho_out;

			ru = rho_out * ux;

			data.f1[offset] = data.f3[offset] + (2./3.)*ru;
			data.f5[offset] = data.f7[offset] + (1./6.)*ru + 0.5*(data.f4[offset] - data.f2[offset]);
			data.f8[offset] = data.f6[offset] + (1./6.)*ru + 0.5*(data.f2[offset] - data.f4[offset]);
		}*/

		// Velocity Boundary at east (outlet)
		/*else if (x == data.width-1 && data.dev == data.deviceCount-1)
		{
			rho = (data.f0[offset] + data.f2[offset] + data.f4[offset] +
					2.*(data.f1[offset] + data.f5[offset] + data.f8[offset])) /
							(1. + ux_in);

			ru = rho * ux_in;

			data.f3[offset] = data.f1[offset] - (2./3.)*ru;
			data.f7[offset] = data.f5[offset] - (1./6.)*ru + 0.5*(data.f2[offset] - data.f4[offset]);
			data.f6[offset] = data.f8[offset] - (1./6.)*ru + 0.5*(data.f4[offset] - data.f2[offset]);
		}*/

		// Pressure Boundary at east
		else if (x == data.width-1 && data.dev == data.deviceCount-1)
		{
			ux = -1. + (data.f0[offset] + data.f2[offset] + data.f4[offset] +
					2.*(data.f1[offset] + data.f5[offset] + data.f8[offset])) / rho_out;

			ru = rho_out * ux;

			data.f3[offset] = data.f1[offset] - (2./3.)*ru;
			data.f7[offset] = data.f5[offset] - (1./6.)*ru + 0.5*(data.f2[offset] - data.f4[offset]);
			data.f6[offset] = data.f8[offset] - (1./6.)*ru + 0.5*(data.f4[offset] - data.f2[offset]);
		}

		// Calculate macroscopic density
		rho =  data.f0[offset] + data.f1[offset] + data.f2[offset] +
				data.f3[offset] + data.f4[offset] + data.f5[offset] +
				data.f6[offset] + data.f7[offset] + data.f8[offset];

		// Calculate macroscopic velocity in x
		ux =  ( data.f1[offset] - data.f3[offset] + data.f5[offset] -
				data.f6[offset] - data.f7[offset] + data.f8[offset] ) / rho;

		// Calculate macroscopic velocity in y
		uy =  ( data.f2[offset] - data.f4[offset] + data.f5[offset] +
				data.f6[offset] - data.f7[offset] - data.f8[offset] ) / rho;

		ueq_x = ux;
		ueq_y = uy;

		// External forces
    	if (Fx != 0)
    		ueq_x = ux + Fx / (rho * tau_inv);
    	if (Fy != 0)
    		ueq_y = uy + Fy / (rho * tau_inv);

		data.ux[offset] = ux;
		data.uy[offset] = uy;
		data.rho[offset] = rho;

    	// Interaction Force - Multiphase fluid
    	if (G != 0)
    	{
    		data.psi[offset] = 4. * exp(- 200. / rho);
			data.ueq_x[offset] = ueq_x;
			data.ueq_y[offset] = ueq_y;
      	} else
    	{
    		// Calculate square velocity and macroscopic velocity
			ueq_xsq = ueq_x * ueq_x;
			ueq_ysq = ueq_y * ueq_y;
			ueq_sq  = ueq_xsq + ueq_ysq;

			float2RGBA(&plot_data[plot_offset*4], ux, uy, 0., rho);

			// Pre-calculate some values
			w0 = w0*rho;
			w1 = w1*rho;
			w5 = w5*rho;
			ueq_xueq_y5 = +ueq_x+ueq_y;
			ueq_xueq_y6 = -ueq_x+ueq_y;

			// Calculate the collision term and pass the new equilibrium distribution function
			data.f0[offset] = data.f0[offset] - tau_inv*( data.f0[offset] -
					w0*(1.f                                - 1.5f*ueq_sq) );
			data.f1[offset] = data.f1[offset] - tau_inv*( data.f1[offset] -
					w1*(1.f + 3.f*ueq_x    + 4.5f*ueq_xsq        - 1.5f*ueq_sq) );
			data.f2[offset] = data.f2[offset] - tau_inv*( data.f2[offset] -
					w1*(1.f + 3.f*ueq_y    + 4.5f*ueq_ysq        - 1.5f*ueq_sq) );
			data.f3[offset] = data.f3[offset] - tau_inv*( data.f3[offset] -
					w1*(1.f - 3.f*ueq_x    + 4.5f*ueq_xsq        - 1.5f*ueq_sq) );
			data.f4[offset] = data.f4[offset] - tau_inv*( data.f4[offset] -
					w1*(1.f - 3.f*ueq_y    + 4.5f*ueq_ysq        - 1.5f*ueq_sq) );
			data.f5[offset] = data.f5[offset] - tau_inv*( data.f5[offset] -
					w5*(1.f + 3.f*ueq_xueq_y5 + 4.5f*ueq_xueq_y5*ueq_xueq_y5 - 1.5f*ueq_sq) );
			data.f6[offset] = data.f6[offset] - tau_inv*( data.f6[offset] -
					w5*(1.f + 3.f*ueq_xueq_y6 + 4.5f*ueq_xueq_y6*ueq_xueq_y6 - 1.5f*ueq_sq) );
			data.f7[offset] = data.f7[offset] - tau_inv*( data.f7[offset] -
					w5*(1.f - 3.f*ueq_xueq_y5 + 4.5f*ueq_xueq_y5*ueq_xueq_y5 - 1.5f*ueq_sq) );
			data.f8[offset] = data.f8[offset] - tau_inv*( data.f8[offset] -
					w5*(1.f - 3.f*ueq_xueq_y6 + 4.5f*ueq_xueq_y6*ueq_xueq_y6 - 1.5f*ueq_sq) );
    	}
	}
	else
	{
		float2RGBA(&plot_data[offset*4], 0, 0, 0, 0);
		plot_data[plot_offset2*4] = 0;
		data.ux[offset] = 0;
		data.uy[offset] = 0;
		data.rho[offset] = 0;
	}
}

extern "C"
void collision_d2q9(float *plot_data, DataD2Q9 *data, float w0, float w1,
		float w5, float tau_inv, float ux_in, float uy_in, float rho_out,
		float u_min, float u_max,float rho_min, float rho_max, dim3 grid, dim3 block,
		float Fx, float Fy, float G, float Gads)
{
	collision_d2q9_kernel<<<grid, block>>> (plot_data, *data, w0, w1, w5, tau_inv,
			ux_in, uy_in, rho_out, u_min, u_max, rho_min, rho_max, Fx, Fy, G);

	if (G < 0)
		multiphase_collision_d2q9_kernel<<<grid, block>>>(plot_data, *data, w0, w1, w5, tau_inv,
				ux_in, uy_in, rho_out, u_min, u_max, rho_min, rho_max, Fx, Fy, G, Gads);
}

__global__ void collision_d3q19_kernel (float *plot_data, DataD3Q19 data,
		float w0, float w1,	float w7, float tau_inv, float ux_in, float uy_in,
		float uz_in, float rho_out,	float u_min, float u_max,
		float rho_min, float rho_max)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = x + y*blockDim.x*gridDim.x + z*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
	//int plot_offset2 = offset +  blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	if (!data.solid[offset])
	{
		// Create registers
		float rho, ux, uy, uz, usq, uxsq, uysq, uzsq, ru;
		float uxuy7, uxuy8, uxuz9, uxuz10, uyuz15, uyuz16;
		float temp;

		//Apply Bounce-Back boundary condition
		temp = data.f1[offset];
		data.f1[offset] = data.f2[offset];
		data.f2[offset] = temp;

		temp = data.f3[offset];
		data.f3[offset] = data.f4[offset];
		data.f4[offset] = temp;

		temp = data.f5[offset];
		data.f5[offset] = data.f6[offset];
		data.f6[offset] = temp;

		temp = data.f7[offset];
		data.f7[offset] = data.f12[offset];
		data.f12[offset] = temp;

		temp = data.f8[offset];
		data.f8[offset] = data.f11[offset];
		data.f11[offset] = temp;

		temp = data.f9[offset];
		data.f9[offset] = data.f14[offset];
		data.f14[offset] = temp;

		temp = data.f10[offset];
		data.f10[offset] = data.f13[offset];
		data.f13[offset] = temp;

		temp = data.f15[offset];
		data.f15[offset] = data.f18[offset];
		data.f18[offset] = temp;

		temp = data.f16[offset];
		data.f16[offset] = data.f17[offset];
		data.f17[offset] = temp;

		// Velocity Boundary at west
		if (x == 0 && data.dev == 0)
		{
			rho = (data.f0[offset] + data.f3[offset] + data.f4[offset] + data.f5[offset] + data.f6[offset] +
				   data.f15[offset] + data.f16[offset] + data.f17[offset] + data.f18[offset] +
				  2.*(data.f2[offset] + data.f11[offset] + data.f12[offset] + data.f13[offset] + data.f14[offset])) /
				  (1. - ux_in);
			ru = rho*ux_in;

			data.f1 [offset] = data.f2 [offset] + (1./3.)*ru;
			data.f8 [offset] = data.f11[offset] + (1./6.)*ru + 0.5*(data.f3[offset] + data.f15[offset] + data.f16[offset] -
			                                                        data.f4[offset] - data.f17[offset] - data.f18[offset]);
			data.f7 [offset] = data.f12[offset] + (1./6.)*ru - 0.5*(data.f3[offset] + data.f15[offset] + data.f16[offset] -
			                                                        data.f4[offset] - data.f17[offset] - data.f18[offset]);
			data.f9 [offset] = data.f14[offset] + (1./6.)*ru - 0.5*(data.f5[offset] + data.f11[offset] + data.f15[offset] -
			                                                        data.f6[offset] - data.f16[offset] - data.f18[offset]);
			data.f10[offset] = data.f13[offset] + (1./6.)*ru + 0.5*(data.f5[offset] + data.f11[offset] + data.f15[offset] -
			                                                        data.f6[offset] - data.f16[offset] - data.f18[offset]);
		}

		// Pressure Boundary at east
		else if (x == data.width-1 && data.dev == data.deviceCount-1)
		{
			ux = -1 + (data.f0[offset] + data.f3[offset] + data.f4[offset] + data.f5[offset] + data.f6[offset] +
				 data.f15[offset] + data.f16[offset] + data.f17[offset] + data.f18[offset] +
				 2.*(data.f1[offset] + data.f7[offset] + data.f8[offset] + data.f9[offset] + data.f10[offset])) /
				 rho_out;
			ru = rho_out*ux;


			data.f2 [offset] = data.f1 [offset] - (1./3.)*ru;
			data.f11[offset] = data.f8 [offset] - (1./6.)*ru - 0.5*(data.f3[offset] + data.f15[offset] + data.f16[offset] -
			                                                        data.f4[offset] - data.f17[offset] - data.f18[offset]);
			data.f12[offset] = data.f7 [offset] - (1./6.)*ru + 0.5*(data.f3[offset] + data.f15[offset] + data.f16[offset] -
			                                                        data.f4[offset] - data.f17[offset] - data.f18[offset]);
			data.f14[offset] = data.f9 [offset] - (1./6.)*ru + 0.5*(data.f5[offset] + data.f11[offset] + data.f15[offset] -
			                                                        data.f6[offset] - data.f16[offset] - data.f18[offset]);
			data.f13[offset] = data.f10[offset] - (1./6.)*ru - 0.5*(data.f5[offset] + data.f11[offset] + data.f15[offset] -
			                                                        data.f6[offset] - data.f16[offset] - data.f18[offset]);
		}

		// Calculate macroscopic density
		rho =  data.f0[offset] + data.f1[offset] + data.f2[offset] +
				data.f3[offset] + data.f4[offset] + data.f5[offset] +
				data.f6[offset] + data.f7[offset] + data.f8[offset] +
				data.f9[offset] + data.f10[offset] + data.f11[offset] +
				data.f12[offset] + data.f13[offset] + data.f14[offset] +
				data.f15[offset] + data.f16[offset] + data.f17[offset] +
				data.f18[offset];

		// Calculate macroscopic velocity in x
		ux =  ( data.f1 [offset] + data.f7 [offset] + data.f8 [offset] +
				data.f9 [offset] + data.f10[offset] - data.f2 [offset] -
				data.f11[offset] - data.f12[offset] - data.f13[offset] -
				data.f14[offset] ) / rho;

		// Calculate macroscopic velocity in y
		uy =  ( data.f3 [offset] + data.f7 [offset] + data.f11[offset] +
				data.f15[offset] + data.f16[offset] - data.f4 [offset] -
				data.f8 [offset] - data.f12[offset] - data.f17[offset] -
				data.f18[offset] ) / rho;

		// Calculate macroscopic velocity in z
		uz =  ( data.f5 [offset] + data.f9 [offset] + data.f13[offset] +
				data.f15[offset] + data.f17[offset] - data.f6 [offset] -
				data.f10[offset] - data.f14[offset] - data.f16[offset] -
				data.f18[offset] ) / rho;

		// Calculate square velocity and macroscopic velocity
		uxsq = ux*ux;
		uysq = uy*uy;
		uzsq = uz*uz;
		usq  = uxsq + uysq + uzsq;
		data.ux[offset] = ux;
		data.uy[offset] = uy;
		data.uz[offset] = uz;
		data.rho[offset] = rho;

		float2RGBA(&plot_data[offset*4], ux, uy, uz, rho);

		// Pre-calculate some values
		w0 = w0*rho;
		w1 = w1*rho;
		w7 = w7*rho;

		uxuy7  = +ux +uy;
		uxuy8  = +ux -uy;
		uxuz9  = +ux +uz;
		uxuz10 = +ux -uz;
		uyuz15 = +uy +uz;
		uyuz16 = +uy -uz;

		data.f0 [offset] = data.f0 [offset] - tau_inv*( data.f0 [offset] -
				w0*(1.f                                   - 1.5f*usq) );
		data.f1 [offset] = data.f1 [offset] - tau_inv*( data.f1 [offset] -
				w1*(1.f + 3.f*ux     + 4.5f*uxsq          - 1.5f*usq) );
		data.f2 [offset] = data.f2 [offset] - tau_inv*( data.f2 [offset] -
				w1*(1.f - 3.f*ux     + 4.5f*uxsq          - 1.5f*usq) );
		data.f3 [offset] = data.f3 [offset] - tau_inv*( data.f3 [offset] -
				w1*(1.f + 3.f*uy     + 4.5f*uysq          - 1.5f*usq));
		data.f4 [offset] = data.f4 [offset] - tau_inv*( data.f4 [offset] -
				w1*(1.f - 3.f*uy     + 4.5f*uysq          - 1.5f*usq) );
		data.f5 [offset] = data.f5 [offset] - tau_inv*( data.f5 [offset] -
				w1*(1.f + 3.f*uz     + 4.5f*uzsq          - 1.5f*usq) );
		data.f6 [offset] = data.f6 [offset] - tau_inv*( data.f6 [offset] -
				w1*(1.f - 3.f*uz     + 4.5f*uzsq          - 1.5f*usq) );
		data.f7 [offset] = data.f7 [offset] - tau_inv*( data.f7 [offset] -
				w7*(1.f + 3.f*uxuy7  + 4.5f*uxuy7 *uxuy7  - 1.5f*usq) );
		data.f8 [offset] = data.f8 [offset] - tau_inv*( data.f8 [offset] -
				w7*(1.f + 3.f*uxuy8  + 4.5f*uxuy8 *uxuy8  - 1.5f*usq) );
		data.f9 [offset] = data.f9 [offset] - tau_inv*( data.f9 [offset] -
				w7*(1.f + 3.f*uxuz9  + 4.5f*uxuz9 *uxuz9  - 1.5f*usq) );
		data.f10[offset] = data.f10[offset] - tau_inv*( data.f10[offset] -
				w7*(1.f + 3.f*uxuz10 + 4.5f*uxuz10*uxuz10 - 1.5f*usq) );
		data.f11[offset] = data.f11[offset] - tau_inv*( data.f11[offset] -
				w7*(1.f - 3.f*uxuy8  + 4.5f*uxuy8 *uxuy8  - 1.5f*usq) );
		data.f12[offset] = data.f12[offset] - tau_inv*( data.f12[offset] -
				w7*(1.f - 3.f*uxuy7  + 4.5f*uxuy7 *uxuy7  - 1.5f*usq) );
		data.f13[offset] = data.f13[offset] - tau_inv*( data.f13[offset] -
				w7*(1.f - 3.f*uxuz10 + 4.5f*uxuz10*uxuz10 - 1.5f*usq) );
		data.f14[offset] = data.f14[offset] - tau_inv*( data.f14[offset] -
				w7*(1.f - 3.f*uxuz9  + 4.5f*uxuz9 *uxuz9  - 1.5f*usq) );
		data.f15[offset] = data.f15[offset] - tau_inv*( data.f15[offset] -
				w7*(1.f + 3.f*uyuz15 + 4.5f*uyuz15*uyuz15 - 1.5f*usq) );
		data.f16[offset] = data.f16[offset] - tau_inv*( data.f16[offset] -
				w7*(1.f + 3.f*uyuz16 + 4.5f*uyuz16*uyuz16 - 1.5f*usq) );
		data.f17[offset] = data.f17[offset] - tau_inv*( data.f17[offset] -
				w7*(1.f - 3.f*uyuz16 + 4.5f*uyuz16*uyuz16 - 1.5f*usq) );
		data.f18[offset] = data.f18[offset] - tau_inv*( data.f18[offset] -
				w7*(1.f - 3.f*uyuz15 + 4.5f*uyuz15*uyuz15 - 1.5f*usq) );

	}
	else
	{
		float2RGBA(&plot_data[offset*4], 0, 0, 0, 0);
		data.ux[offset] = 0;
		data.uy[offset] = 0;
		data.uz[offset] = 0;
		data.rho[offset] = 0;
	}
}

// C collision function that calls CUDA kernel
extern "C"
void collision_d3q19(float *plot_data, DataD3Q19 *data, float w0, float w1,
		float w7, float tau_inv, float ux_in, float uy_in, float uz_in, float rho_out,
		float u_min, float u_max,float rho_min, float rho_max, dim3 grid, dim3 block)
{
	collision_d3q19_kernel<<<grid, block>>> (plot_data, *data, w0, w1, w7, tau_inv,
			ux_in, uy_in, uz_in, rho_out, u_min, u_max, rho_min, rho_max);
}
