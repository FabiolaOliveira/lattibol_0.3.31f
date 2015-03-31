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

#ifndef SIMULATION_H
#define SIMULATION_H

#include <cuda_gl_interop.h>
#include "input.h"
#include "output.h"
#include "lattibol.h"

#define BLOCK_X 32
#define BLOCK_Y 4
#define BLOCK_Z 1

class Simulation
{
public:
	Simulation(Input *inputPtr);

	virtual void runLBM() = 0;
	virtual void saveVtk() = 0;

	Output* getOutput();
	int getDeviceCount();

	int getWidth();
	int getHeight();
	int getDepth();
	Cursor* getCursor();

	float getMinU();
	float getMaxU();
	float getMinRho();
	float getMaxRho();

	unsigned int getSteps();

	void initCudaGL(GLuint pbo, int dev);

protected:
	Input *input;
	Output *output;

	int deviceCount, commSize;

	struct cudaGraphicsResource **cuda_pbo_resource; // handles OpenGL-CUDA exchange

	dim3 grid;
	dim3 block;

	unsigned int width;
	unsigned int height;
	unsigned int depth;
	unsigned int size;

	float w0;
	float w1;
	float w2;

	float **plotData;
	size_t num_bytes;

	Cursor *cursor;

	unsigned int steps;
};

#endif
