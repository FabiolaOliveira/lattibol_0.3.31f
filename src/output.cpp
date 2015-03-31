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

#include <fstream>
#include <iostream>
#include "mpi.h"
#include "output.h"
#include "input.h"

using namespace std;

Output::Output(Input input) {
	deviceCount = input.getDeviceCount();
	width = input.getWidth()/deviceCount;
	height = input.getHeight();
	length = input.getLength();

	updateSteps = input.getUpdateSteps();

	vtkId = 0;

	setTimeStart();
}

void Output::setUpdateSteps(int value) { updateSteps = value; }

void Output::setData(float *ux, float *uy, float *uz, float *rho) {
	dataUx = ux;
	dataUy = uy;
	dataUz = uz;
	dataRho = rho;
}

void Output::saveParVTK () {

	char name[13];

	// Get the name 'rxxxx.pvti' of save file according to the number of the vtkId
	name[0] = 'r';
	name[1] = (char)(vtkId / 1000) + '0';
	name[2] = (char)(vtkId % 1000 / 100) + '0';
	name[3] = (char)(vtkId % 100 / 10) + '0';
	name[4] = (char)(vtkId % 10) + '0';
	name[5] = '.';
	name[6] = 'p';
	name[7] = 'v';
	name[8] = 't';
	name[9] = 'i';
	name[10] = '\0';

	// Create and open 'rxxxx.pvti' file
	ofstream file(name, ofstream::out);

	// Change name string to name of the dataset file
	name[5] = 'p';
	name[8] = '.';
	name[9] = 'v';
	name[10] = 't';
	name[11] = 'i';
	name[12] = '\0';

	// Save the parallel file according to VTK file format
	file << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
			<< "  <PImageData WholeExtent=\"" << 0 << " " << width*deviceCount-1 << " 0 " << height-1
			<< " 0 " << length-1 << "\" GhostLevel=\"#\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
			<< "    <PPointData Scalars=\"Pressure\" Vectors=\"Velocity\">\n"
			<< "      <DataArray type=\"Float32\" Name=\"Pressure\" format=\"ascii\"/>\n"
			<< "      <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" "
			<< "format=\"ascii\"/>\n    </PPointData>\n    <PCellData>\n    </PCellData>\n";
	for (int i=0; i < deviceCount; i++) {
		name[6] = (char)(i / 10) + '0';
		name[7] = (char)(i % 10) + '0';
		file << "    <Piece Extent=\"" << i*width << " " << (i+1)*width << " 0 " << height-1 << " 0 " << length-1
				<< "\" Source=\"" <<	name << "\"/>\n";
	}
	file << "  </PImageData>\n</VTKFile>\n";
	file.close();
}

void Output::saveVTK (int deviceID) {

	char name[13];
	int i;

	// Get the name 'rxxxxpxx.vti' of save file according to the number of the
	// vtkId and the process
	name[0] = 'r';
	name[1] = (char)(vtkId / 1000) + '0';
	name[2] = (char)(vtkId % 1000 / 100) + '0';
	name[3] = (char)(vtkId % 100 / 10) + '0';
	name[4] = (char)(vtkId % 10) + '0';
	name[5] = 'p';
	name[6] = (char)(deviceID / 10) + '0';
	name[7] = (char)(deviceID % 10) + '0';
	name[8] = '.';
	name[9] = 'v';
	name[10] = 't';
	name[11] = 'i';
	name[12] = '\0';

	// Create and open 'rxxxxpxx.vti' file
	ofstream file(name, ofstream::out);

	// Save dataset file according to VTK file format
	file << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
	<< "  <ImageData WholeExtent=\"" << deviceID*width <<" " << (deviceID+1)*width << " 0 " << height-1 << " 0 " << length-1
	<< "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
	<< "  <Piece Extent=\"" << deviceID*width << " " << (deviceID+1)*width << " 0 " << height-1 << " 0 " << length-1 << "\">\n"
	<< "    <PointData Scalars=\"Pressure\" Vectors=\"Velocity\">\n"
	<< "      <DataArray type=\"Float32\" Name=\"Pressure\" format=\"ascii\" >\n      ";

	// Save dataset to file.
	// If the lattice node is solid, the save data is 0.0, else, save data is the velocity
	for(i=0; i < width*height*length; i++)	{
		file << dataRho[i] << " ";
		if ((i+1) % width == 0) file << "0\n      ";
	}

	file << "</DataArray>\n"
	<< "      <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n"
	<< "      ";

	// Save dataset to file.
	// If the lattice node is solid, the save data is 0.0, else, save data is the velocity
	for(i=0; i < width*height*length; i++)	{
		file << dataUx[i] << " ";
		if(length > 1) file << dataUy[i] << " "<< dataUz[i] << "   ";
		else file << -dataUy[i] << " 0   ";
		if ((i+1) % width == 0) file << "0 0 0\n      ";
	}

	file << "</DataArray>\n    </PointData>\n    <CellData>\n    "
			"</CellData>\n  </Piece>\n  </ImageData>\n</VTKFile>\n";

	file.close();
	if(deviceID == deviceCount-1) vtkId++;
}

void Output::setTimeStart()
{
	MPI_Barrier(MPI_COMM_WORLD);
	startTime = MPI_Wtime();
	initPauseTime();
}

double Output::getTime()
{
	MPI_Barrier(MPI_COMM_WORLD);
	return MPI_Wtime() - startTime;
}

void Output::initPauseTime() { pauseTime = getTime(); }
void Output::setLBMTime() { lbmTime = getTime() - lbmTime; }
double Output::getMLUPS() { return 1e-6*updateSteps*width*height*length/lbmTime; }

void Output::initLBMTime() { lbmTime = getTime(); }
void Output::setPauseTime() { startTime += getTime() - pauseTime; }

void Output::initFPSTime() { fpsTime = getTime(); }
double Output::getFPS() { return 1./(getTime() - fpsTime); }
