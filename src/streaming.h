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
 *																		   *
 * This header file defines the streaming function. This functions passes  *
 * the actual density functions to the nearest lattice node in 'fx_new'    *
 * according to its direction.											   *
 ***************************************************************************/

#ifndef STREAMING_H
#define STREAMING_H

#include "lattibol.h"

extern "C" 
void streaming_d2q9( DataD2Q9 *data, dim3 grid, dim3 block );

extern "C"
void streaming_d3q19( DataD3Q19 *data, dim3 grid, dim3 block );

#endif
