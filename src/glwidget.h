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

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <QtOpenGL>
#include "simulation.h"

class GLWidget : public QGLWidget
{
    Q_OBJECT

public:
    GLWidget();
    ~GLWidget();

    virtual void clipX(int pos, bool dir) = 0;
    virtual void clipY(int pos, bool dir) = 0;
    virtual void clipZ(int pos, bool dir) = 0;

    bool loadColormap(const QString &fileName);
    void setVelocityScale(float min, float max);
    void setDensityScale(float min, float max);

    void setShader(int index);

protected:
    unsigned int colormap[256];

    float minU, maxU, minRho, maxRho;

    Cursor *cursor;

    GLuint *pbo;     			// OpenGL pixel buffer object
    GLuint texData, texTF;    // textures

    int currentShader;
    QVector<QGLShaderProgram *> shaderPrograms;
    QVector<QGLShader *> fragmentShaders;
};

#endif
