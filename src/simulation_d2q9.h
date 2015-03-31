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

#ifndef SIMULATION_D2Q9_H
#define SIMULATION_D2Q9_H

#include <cuda_gl_interop.h>
#include "lattibol.h"
#include "simulation.h"
#include "input.h"
#include "output.h"

class SimulationD2Q9 : public Simulation
{
public:
	SimulationD2Q9(Input* inputPtr);

	void runLBM();
	void saveVtk();

private:
	DataD2Q9 *data;
};

#endif
