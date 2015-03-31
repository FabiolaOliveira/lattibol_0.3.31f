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

#ifndef GLWIDGET_2D_H
#define GLWIDGET_2D_H

#include <QGLWidget>
#include <QtOpenGL>
#include "glwidget.h"
#include "simulation.h"

class GLWidget2D: public GLWidget
{
    Q_OBJECT

public:
    GLWidget2D(Simulation *sim = 0);
    ~GLWidget2D();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    void clipX(int pos, bool dir);
    void clipY(int pos, bool dir);
    void clipZ(int pos, bool dir);

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);

private:
    GLuint *pbo;     // OpenGL pixel buffer object
    GLuint texid;   // texture
    int width;
    int height;
    int deviceCount;
    Simulation *simulation;
};

#endif
