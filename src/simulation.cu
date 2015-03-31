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

#include "simulation.h"

//Simulation::Simulation() {;}

Simulation::Simulation(Input* inputPtr) {
    input = inputPtr;
	output = new Output(*input);
	deviceCount = input->getDeviceCount();
	commSize = 1;

	cuda_pbo_resource = new cudaGraphicsResource*[deviceCount];
	num_bytes = 0;

	width = input->getWidth()/deviceCount;
	height = input->getHeight();
	depth = input->getLength();

	size = width*height*depth;

	plotData = new float*[deviceCount];

	grid = dim3(width/BLOCK_X, height/BLOCK_Y, depth/BLOCK_Z);
	block = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);

	steps = 0;
}

int Simulation::getWidth() { return input->getWidth(); }
int Simulation::getHeight() { return input->getHeight(); }
int Simulation::getDepth() { return input->getLength(); }
int Simulation::getDeviceCount() { return input->getDeviceCount(); }
Cursor* Simulation::getCursor() { cursor = input->getCursor(); return cursor;}

void Simulation::initCudaGL(GLuint pbo, int dev) {
	cudaSetDevice(0);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource[dev], pbo, cudaGraphicsMapFlagsWriteDiscard);
}

Output* Simulation::getOutput() { return output; }

float Simulation::getMinU() {return input->getMinU();}
float Simulation::getMaxU() {return input->getMaxU();}
float Simulation::getMinRho() {return input->getMinRho();}
float Simulation::getMaxRho() {return input->getMaxRho();}

unsigned int Simulation::getSteps() {return steps;}
