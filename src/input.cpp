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
 ***************************************************************************/

#include "input.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <cstring>

#include <QWidget>
#include <QObject>
#include <QFile>
#include <QMessageBox>
#include <QTextStream>
#include <QObject>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>

using namespace cvmlcpp;
using namespace std;

bool Input::loadParameters(const QString &fileName)
{
	QFile file(fileName);
	if (!file.open(QFile::ReadOnly | QFile::Text)) return 0;

    QTextStream in(&file);
	QString line = in.readLine();
	setDeviceCount(line.toInt());
	line = in.readLine();
	setMaxDim(line.toInt());
	line = in.readLine();
	setUx(line.toFloat());
	line = in.readLine();
	setUy(line.toFloat());
	line = in.readLine();
	setUz(line.toFloat());
	line = in.readLine();
	setRho(line.toFloat());
	line = in.readLine();
	setViscosity(line.toFloat());
	line = in.readLine();
	setUpdateSteps(line.toInt());

	// External forces
	line = in.readLine();
	setExtFx(line.toFloat());
	line = in.readLine();
	setExtFy(line.toFloat());
	line = in.readLine();
	setExtFz(line.toFloat());
	// End - External forces

	// Multiphase fluid
	line = in.readLine();
	setG(line.toFloat());

	// Fluid-Surface force
	line = in.readLine();
	setGads(line.toFloat());

	setMinU(0.00);
	setMaxU(uxIn*1.5);
	setMinRho(rhoOut*0.98);
	setMaxRho(rhoOut*1.02);

	return 1;
}

bool Input::loadSolids(const QString &fileName)
{
    QImage image;
	int x, y;

	if (!image.load(fileName)) return 0;

	width = image.width();
	height = image.height();

	solids = new bool[width*height];

	for(y=0; y < height; y++) for(x=0; x < width; x++)
		if(image.pixel(x, y) == qRgb(0,0,0)) solids[x+y*width] = 1;
		else solids[x+y*width] = 0;

	return 1;
}

bool Input::loadSolidsSTL(const QString &fileName)
{
	Matrix<char, 3u> voxels;
	Geometry<float> geometry;

	int x, y, z;
	double voxelSize = 1.0 / (float)maxDim;

	readSTL(geometry, fileName.toStdString());

	geometry.scaleTo(1.0);
	voxelize(geometry, voxels, voxelSize, 1 /* pad */, (char)1 /* inside */, (char)0 /*outside */);

	width = voxels.extents()[X]-2;
	height = voxels.extents()[Y]-2;
	depth = voxels.extents()[Z]-2;

	solids = new bool[width*height*depth];

	for (x = 0; x < width; x++)
		for (y = 0; y < height; y++)
			for (z = 0; z < depth; z++)
				solids[x + y*width + z*width*height] = voxels[x+1][y+1][z+1];
	return 1;
}

int Input::getWidth() { return width; }
int Input::getHeight() { return height; }
int Input::getLength() { return depth; }

int Input::getMaxDim() { return maxDim; }
void Input::setMaxDim(int value) {
	if(value > 1) maxDim = value;
	else { maxDim = 1; depth = 1; }
}

float Input::getUx() {	return uxIn; }
void Input::setUx(float value) { uxIn = value; }

float Input::getUy() {	return uyIn; }
void Input::setUy(float value) { uyIn = value; }

float Input::getUz() {	return uzIn; }
void Input::setUz(float value) { uzIn = value; }

float Input::getRho() { return rhoOut; }
void Input::setRho(float value) { rhoOut = value; }

float Input::getViscosity() { return viscosity; }
float Input::getTauInv() {	return tauInv; }
void Input::setViscosity(float value) {
	viscosity = value;
	tauInv = 1./(3.*viscosity + 0.5);
}

int Input::getUpdateSteps() { return updateSteps; }
void Input::setUpdateSteps(int value) { updateSteps = value; }

// External forces
float Input::getExtFx() { return extFx; }
void Input::setExtFx(float value) { extFx = value; }

float Input::getExtFy() { return extFy; }
void Input::setExtFy(float value) { extFy = value; }

float Input::getExtFz() { return extFz; }
void Input::setExtFz(float value) { extFz = value; }
// End external forces

// Interaction force (multiphase fluids)
float Input::getG() { return G; }
void Input::setG(float value) { G = value; }
// End - Interaction force

// Adhesion force (multiphase fluids/fluid-surface forces)
float Input::getGads() { return Gads; }
void Input::setGads(float value) { Gads = value; }
// End - Interaction force

bool* Input::getSolids() {	return solids; }

float Input::getMinU() { return minU; }
void Input::setMinU(float value) { minU = value; }

float Input::getMaxU() { return maxU; }
void Input::setMaxU(float value) { maxU = value; }

float Input::getMinRho() { return minRho; }
void Input::setMinRho(float value) { minRho = value; }

float Input::getMaxRho() { return maxRho; }
void Input::setMaxRho(float value) { maxRho = value; }

int Input::getDeviceCount() { return deviceCount; }
void Input::setDeviceCount(int value) { deviceCount = value; }

Cursor* Input::getCursor() { cursor = new Cursor(width, height, depth); return cursor; }

void Input::setCursorValue(float *value) {
	uxCursor = value[0];
	uyCursor = value[1];
	uzCursor = value[2];
	uMagCursor = sqrt(pow(value[0],2)+pow(value[1],2)+pow(value[2],2));
	rhoCursor = value[3];
}

void Input::setCursorX(int value) {cursor->setX(value);}
void Input::setCursorY(int value) {cursor->setY(value);}
void Input::setCursorZ(int value) {cursor->setZ(value);}

int Input::getCursorX() {return cursor->getX();}
int Input::getCursorY() {return cursor->getY();}
int Input::getCursorZ() {return cursor->getZ();}

float Input::getCursorUX() {return uxCursor;}
float Input::getCursorUY() {return uyCursor;}
float Input::getCursorUZ() {return uzCursor;}
float Input::getCursorUMag() {return uMagCursor;}
float Input::getCursorRho() {return rhoCursor;}
