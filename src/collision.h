/***************************************************************************
 *   Copyright (C) 2014 by Lucas Monteiro Volpe             		       *
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

#ifndef COLLISION_H
#define COLLISION_H

#include "lattibol.h"

extern "C"
void collision_d2q9(float *plot_data, DataD2Q9 *data, float w0, float w1,
		float w5, float tau_inv, float ux_in, float uy_in, float rho_out,
		float u_min, float u_max,float rho_min, float rho_max, dim3 grid, dim3 block,
		float Fx, float Fy, float G, float Gads);

extern "C"
void collision_d3q19(float *plot_data, DataD3Q19 *data, float w0, float w1,
		float w7, float tau_inv, float ux_in, float uy_in, float uz_in, float rho_out,
		float u_min, float u_max,float rho_min, float rho_max, dim3 grid, dim3 block);

#endif
