/***************************************************************************
 *   Copyright (C) 2013 by Lucas Monteiro Volpe             		       *
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

#ifndef OUTPUT_H
#define OUTPUT_H

#include "input.h"

class Output
{
public:

	Output(Input input);

	void saveVTK (int deviceID);
	void saveParVTK ();

	void setData(float *ux, float *uy, float *uz, float *rho);

	void setUpdateSteps(int value);

	void setTimeStart();
	double getTime();

	void initLBMTime();
	void setLBMTime();
	double getMLUPS();

	void initPauseTime();
	void setPauseTime();

	void initFPSTime();
	void setFPSTime();
	double getFPS();

private:
	// Data
	float *dataUx;
	float *dataUy;
	float *dataUz;
	float *dataRho;

	int vtkId;
	int updateSteps;
	int deviceCount;
	int deviceID;

	// Size of the domain
	int width, height, length;

	double startTime;
	double lbmTime;
	double pauseTime;
	double fpsTime;
};

#endif
