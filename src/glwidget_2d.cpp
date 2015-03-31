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

#include <math.h>
#include <mpi.h>

#include "glwidget_2d.h"
#include "simulation.h"

#include <iostream>
using namespace std;

GLWidget2D::GLWidget2D(Simulation *sim)
{
	format().setSwapInterval(0);
	simulation = sim;

	deviceCount = simulation->getDeviceCount();
	width = simulation->getWidth()/deviceCount;
	height = simulation->getHeight()*2;
	pbo = new GLuint[deviceCount];
	cursor = simulation->getCursor();
	texData = 0;

    minU = 0.0;
    maxU = 0.015;
    minRho = 0.98;
    maxRho = 1.20;

    currentShader = 0;
}

GLWidget2D::~GLWidget2D()
{
}


QSize GLWidget2D::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GLWidget2D::sizeHint() const
{
    return QSize(width*deviceCount, height);
}

void GLWidget2D::initializeGL()
{
    QStringList filter;
    QList<QFileInfo> files;

    filter = QStringList("*.fsh");
    files = QDir("./shaders/2d/").entryInfoList(filter, QDir::Files | QDir::Readable);
    foreach (QFileInfo file, files) {
        QGLShaderProgram *program = new QGLShaderProgram;
        QGLShader* shader = new QGLShader(QGLShader::Fragment);
        shader->compileSourceFile(file.absoluteFilePath());

        // The program does not take ownership over the shaders, so store them in a vector so they can be deleted afterwards.
        program->addShader(shader);
        if (!program->link()) {
            qWarning("Failed to compile and link shader program");
            qWarning() << "Fragment shader log ( file =" << file.absoluteFilePath() << "):";
            qWarning() << shader->log();
            qWarning("Shader program log:");
            qWarning() << program->log();

            delete shader;
            delete program;
            continue;
        }
        fragmentShaders << shader;
        shaderPrograms << program;
    }

    if (shaderPrograms.size() == 0)
        shaderPrograms << new QGLShaderProgram;

    glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// create pixel buffer object
	glGenBuffersARB(deviceCount, pbo);
	for(int dev = 0; dev < deviceCount; dev++) {
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo[dev]);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLfloat)*4, NULL, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		simulation->initCudaGL(pbo[dev], dev);
	}

    // create texture for display
	glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texData);
    glBindTexture(GL_TEXTURE_2D, texData);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width*deviceCount, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &texTF);
    glBindTexture(GL_TEXTURE_1D, texTF);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_1D, 0);
    glDisable(GL_TEXTURE_1D);

	loadColormap("./colormaps/blue_to_red.png");
}

void GLWidget2D::paintGL()
{
	minU = simulation->getMinU();
	maxU = simulation->getMaxU();
	minRho = simulation->getMinRho();
	maxRho = simulation->getMaxRho();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shaderPrograms[currentShader]->bind();

    // load 1d texture
	glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, texTF);
    shaderPrograms[currentShader]->setUniformValue("COLOR", 0);

	// load texture from pbo
    glActiveTexture(GL_TEXTURE1);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texData);
	for(int dev = 0; dev < deviceCount; dev++) {
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo[dev]);
		glTexSubImage2D(GL_TEXTURE_2D, 0, dev*width, 0, width, height, GL_RGBA, GL_FLOAT, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	}
	shaderPrograms[currentShader]->setUniformValue("VOLUME", 1);
	shaderPrograms[currentShader]->setUniformValue("minU", minU);
	shaderPrograms[currentShader]->setUniformValue("maxU", maxU);
	shaderPrograms[currentShader]->setUniformValue("minRho", minRho);
	shaderPrograms[currentShader]->setUniformValue("maxRho", maxRho);

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0, 0);
		glVertex2f(-1, -1);
		glTexCoord2f(1, 0);
		glVertex2f(1, -1);
		glTexCoord2f(1, 1);
		glVertex2f(1, 1);
		glTexCoord2f(0, 1);
		glVertex2f(-1, 1);
	}
	glEnd();

    shaderPrograms[currentShader]->release();
    glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_1D, 0);
	glDisable(GL_TEXTURE_1D);

	cursor->show();
}

void GLWidget2D::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
}


void GLWidget2D::clipX(int pos, bool dir)
{

}

void GLWidget2D::clipY(int pos, bool dir)
{

}

void GLWidget2D::clipZ(int pos, bool dir)
{

}
