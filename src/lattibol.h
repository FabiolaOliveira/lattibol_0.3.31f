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
 *																		   *
 * This header file defines the memory structure that will be allocated in each device.	*
 * Each variable is an array of the domain total dimension, except the border variables,*
 * that has the domain height dimension as lenght.										*
 ****************************************************************************************/

#ifndef LB_H
#define LB_H

typedef struct {

	// Velocity [lu/ts]
	float *ux, *uy, *rho;
	
	// Solid node
	bool  *solid, *solid_bound;

	// Density functions
	float *f0, *f1, *f2, *f3, *f4;
	float *f5, *f6, *f7, *f8;

	// Border Density Functions
	float *border_f1, *border_f5, *border_f8;
	float *border_f3, *border_f6, *border_f7;

	// Neighbor border density function
	float *nb_border_f1, *nb_border_f5, *nb_border_f8;
	float *nb_border_f3, *nb_border_f6, *nb_border_f7;

	// Size of the domain
	int width, height;

	int deviceCount, dev;
	float *plotBuffer;
	float *valueCur;

	// Stream
    cudaStream_t stream;

    // Multiphase fluid
    float *psi;
    float *ueq_x, *ueq_y;

} DataD2Q9;

typedef struct {

	// Velocity [lu/ts]
	float *ux, *uy, *uz, *rho;

	// Solid node
	bool  *solid, *solid_bound;

	// Density functions
	float *f0, *f1, *f2, *f3, *f4;
	float *f5, *f6, *f7, *f8, *f9;
	float *f10, *f11, *f12, *f13, *f14;
	float *f15, *f16, *f17, *f18;

	// Border Density Functions
	float *border_f1, *border_f7, *border_f8, *border_f9, *border_f10;
	float *border_f2, *border_f11, *border_f12, *border_f13, *border_f14;

	// Neighbor border density function
	float *nb_border_f1, *nb_border_f7, *nb_border_f8, *nb_border_f9, *nb_border_f10;
	float *nb_border_f2, *nb_border_f11, *nb_border_f12, *nb_border_f13, *nb_border_f14;

	// Size of the domain
	int width, height, depth;

	int deviceCount, dev;
	float *plotBuffer;
	float *valueCur;

	// Stream
    cudaStream_t stream;

} DataD3Q19;


#endif
