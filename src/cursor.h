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
 ***************************************************************************/

#ifndef CURSOR_H
#define CURSOR_H

//#include <QGLWidget>
//#include <QtOpenGL>
#include <GL/gl.h>

class Cursor
{
    //Q_OBJECT

public:
    Cursor(int w, int h, int d);
    void show();
    void getPos(int *xPos, int *yPos, int *zPos);
    void move(float xMove, float yMove, float zMove);

    void setX(int value);
    void setY(int value);
    void setZ(int value);
    int getX();
    int getY();
    int getZ();

//protected:

//private slots:

private:
    float x, y, z;
    int space, size;
    int width, height, depth;
    int xOffset, yOffset, zOffset;

};

#endif
