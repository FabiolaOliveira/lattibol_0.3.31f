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
 * This header file defines the copy_border function. This functions copies nb_border_fx to     *
 * border_fx from any device to any device using UVA, so its necessary compute capability 2.0+. *
 * To use this function its necessary to pass the current device nb_border_fx, 'left' device    *
 * border_(f1, f5, f8) and 'right' device border_(f3, f6, f7).								    *
 ***********************************************************************************************/

#ifndef COPY_BORDER_H
#define COPY_BORDER_H

extern "C" 
void copy_border_devices_d2q9(
		float *nb_border_f1, float *nb_border_f5, float *nb_border_f8,
		float *nb_border_f3, float *nb_border_f6, float *nb_border_f7,
		float *border_f1, float *border_f5, float *border_f8,
		float *border_f3, float *border_f6, float *border_f7,
		int height, cudaStream_t &s);

void copy_border_devices_d3q19(
		float *nb_border_f1, float *nb_border_f7, float *nb_border_f8,
		float *nb_border_f9, float *nb_border_f10,
		float *nb_border_f2, float *nb_border_f11, float *nb_border_f12,
		float *nb_border_f13, float *nb_border_f14,
		float *border_f1, float *border_f7, float *border_f8,
		float *border_f9, float *border_f10,
		float *border_f2, float *border_f11, float *border_f12,
		float *border_f13, float *border_f14,
		int height, int depth, cudaStream_t &s);

void copy_border_nodes_d2q9(
		float *border_ctr_buf, float *border_top_buf, float *border_btm_buf,
		float *nb_border_ctr, float *nb_border_top, float *nb_border_btm,
		int height, int dest, int source, cudaStream_t &s);

void copy_border_nodes_d3q19(
		float *border_ctr_buf, float *border_lft_buf, float *border_rgt_buf,
		float *border_top_buf, float *border_btm_buf,
		float *nb_border_ctr, float *nb_border_lft, float *nb_border_rgt,
		float *nb_border_top, float *nb_border_btm,
		int height, int depth, int dest, int source, cudaStream_t &s);

#endif
