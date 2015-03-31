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

#ifndef INPUT_H
#define INPUT_H
#include <iostream>
#include <string>

class QString;

#include "cursor.h"

class Input
{

public:
	bool loadParameters(const QString &fileName);
	bool loadSolids(const QString &fileName);
	bool loadSolidsSTL(const QString &fileName);

	int getWidth();
	int getHeight();
	int getLength();
	void setLength(int value);

	int getMaxDim();
	void setMaxDim(int value);

	float getUx();
	void setUx(float value);

	float getUy();
	void setUy(float value);

	float getUz();
	void setUz(float value);

	float getRho();
	void setRho(float value);

	float getViscosity();
	float getTauInv();
	void setViscosity(float value);

	int getUpdateSteps();
	void setUpdateSteps(int value);

	// External forces
	float getExtFx();
	void setExtFx(float value);

	float getExtFy();
	void setExtFy(float value);

	float getExtFz();
	void setExtFz(float value);
	// End - External forces

	// Interactive force (multiphase fluid)
	float getG();
	void setG(float value);
	// End - interactive force

	// Adhesion force (multiphase fluid/ fluid-surface force)
	float getGads();
	void setGads(float value);
	// End - adhesion force

	bool* getSolids();

	float getMinU();
	void setMinU(float value);

	float getMaxU();
	void setMaxU(float value);

	float getMinRho();
	void setMinRho(float value);

	float getMaxRho();
	void setMaxRho(float value);

	int getDeviceCount();
	void setDeviceCount(int value);

	Cursor* getCursor();

	void setCursorX(int value);
	void setCursorY(int value);
	void setCursorZ(int value);
	void setCursorValue(float *value);

	int getCursorX();
	int getCursorY();
	int getCursorZ();
	float getCursorUX();
	float getCursorUY();
	float getCursorUZ();
	float getCursorUMag();
	float getCursorRho();

private:
	int width;
	int height;
	int depth;

	int maxDim;

	float uxIn;
	float uyIn;
	float uzIn;
	float rhoOut;

	float viscosity, tauInv;

	float minU;
	float maxU;

	float minRho;
	float maxRho;

	int updateSteps;

	int deviceCount;

	// External forces
	float extFx, extFy, extFz;

	// Interactive force
	float G;
	// Adhesion force
	float Gads;

	bool *solids;

	Cursor *cursor;

	int xCur, yCur, zCur;
	float uxCursor, uyCursor, uzCursor;
	float uMagCursor, rhoCursor;

};

#endif
