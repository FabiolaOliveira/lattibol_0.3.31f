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

#define GL_GLEXT_PROTOTYPES

#include <QtWidgets>
#include "glwidget.h"

GLWidget::GLWidget()
    : QGLWidget(QGLFormat(QGL::SampleBuffers ), 0)
{
}

GLWidget::~GLWidget()
{
}

bool GLWidget::loadColormap(const QString &fileName)
{
    QImage image;
	int x;

	if (!image.load(fileName)) return 0;

	for(x=0; x < 256; x++) colormap[x] = image.pixel(x, 0);
	glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, texTF);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, 256, 0, GL_BGRA,
            GL_UNSIGNED_INT_8_8_8_8_REV, colormap);
    glBindTexture(GL_TEXTURE_1D, 0);
	glDisable(GL_TEXTURE_1D);

	return 1;
}

void GLWidget::setVelocityScale(float min, float max) {minU = min; maxU = max;}
void GLWidget::setDensityScale(float min, float max) {minRho = min; maxRho = max;}

void GLWidget::setShader(int index) {currentShader = index;}
