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
 *																		   *
 * This header file defines the get_border function. This functions passes the most right density *
 * functions f1, f5, and f8 from fx_new to border_fx and the most left density functions 		  *
 * f3, f6, and f7 from fx_new to border_fx.														  *
 *************************************************************************************************/

#ifndef GET_BORDER_H
#define GET_BORDER_H

extern "C"
void get_border_d2q9 ( DataD2Q9 *data, dim3 grid, dim3 block );

extern "C"
void get_border_d3q19 ( DataD3Q19 *data, dim3 grid, dim3 block );

#endif
