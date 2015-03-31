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

#include "glwidget_3d.h"
#include "simulation.h"
#include "trackball.h"

#include <iostream>
using namespace std;

GLWidget3D::GLWidget3D(Simulation *sim)
{
	format().setSwapInterval(0);
	simulation = sim;

	deviceCount = simulation->getDeviceCount();
	width = simulation->getWidth()/deviceCount;
	height = simulation->getHeight();
	depth = simulation->getDepth();
	cursor = simulation->getCursor();
	pbo = new GLuint[deviceCount];
	texData = 0;

	maxDim = width;
	if(maxDim < height) maxDim = height;
	if(maxDim < depth) maxDim = depth;
	showAll();

    trackBall = TrackBall(0.0f, QVector3D(0, 1, 0), TrackBall::Plane);

    zoom = 1.0;
    zoomCount = 0;

    xPan = 0;
    yPan = 0;

    minU = 0.0;
    maxU = 0.015;
    minRho = 0.98;
    maxRho = 1.20;

    currentShader = 0;
}

GLWidget3D::~GLWidget3D()
{
}

QSize GLWidget3D::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GLWidget3D::sizeHint() const
{
    return QSize(width*deviceCount, height);
}

void GLWidget3D::initializeGL()
{
    QStringList filter;
    QList<QFileInfo> files;

    filter = QStringList("*.fsh");
    files = QDir("./shaders/3d/").entryInfoList(filter, QDir::Files | QDir::Readable);
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

    glClearColor(0.6, 0.6, 0.7, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

	// create pixel buffer object
	glGenBuffersARB(deviceCount, pbo);
	for(int dev = 0; dev < deviceCount; dev++) {
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo[dev]);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*depth*sizeof(GLfloat)*4, NULL, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		simulation->initCudaGL(pbo[dev], dev);
	}

    // create texture for display
	glEnable(GL_TEXTURE_3D);
    glGenTextures(1, &texData);
    glBindTexture(GL_TEXTURE_3D, texData);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, width*deviceCount, height, depth, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glBindTexture(GL_TEXTURE_3D, 0);
	glDisable(GL_TEXTURE_3D);

	glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &texTF);
    glBindTexture(GL_TEXTURE_1D, texTF);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_1D, 0);
    glDisable(GL_TEXTURE_1D);

	loadColormap("./colormaps/blue_to_red.png");

}

void GLWidget3D::showAll()
{
	xMinTex = 0.5/width;
	xMaxTex = (width-0.5)/width;
	yMinTex = 0.5/height;
	yMaxTex = (height-0.5)/height;
	zMinTex = 0.5/depth;
	zMaxTex = (depth-0.5)/depth;

	xMinVrt = -width/2;
	xMaxVrt = width/2;
	yMinVrt = -height/2;
	yMaxVrt = height/2;
	zMinVrt = -depth/2;
	zMaxVrt = depth/2;
}

void GLWidget3D::clipX(int pos, bool dir)
{
	if(dir == 0) {
		xMinTex = 0.5/width;
		xMaxTex = (pos+0.5)/width;

		xMinVrt = -width/2;
		xMaxVrt = -width/2 + pos;
	}

	if(dir == 1) {
		xMinTex = (pos+0.5)/width;
		xMaxTex = (width-0.5)/width;

		xMinVrt = -width/2 + pos;
		xMaxVrt = width/2;
	}
	updateGL();
}

void GLWidget3D::clipY(int pos, bool dir)
{
	if(dir == 0) {
		yMinTex = 0.5/height;
		yMaxTex = (pos+0.5)/height;

		yMinVrt = -height/2;
		yMaxVrt = -height/2 + pos;
	}

	if(dir == 1) {
		yMinTex = (pos+0.5)/height;
		yMaxTex = (height-0.5)/height;

		yMinVrt = -height/2 + pos;
		yMaxVrt = height/2;
	}
	updateGL();
}

void GLWidget3D::clipZ(int pos, bool dir)
{
	if(dir == 0) {
		zMinTex = 0.5/depth;
		zMaxTex = (pos+0.5)/depth;

		zMinVrt = depth/2 - pos;
		zMaxVrt = depth/2;
	}

	if(dir == 1) {
		zMinTex = (pos+0.5)/depth;
		zMaxTex = (depth-0.5)/depth;

		zMinVrt = -depth/2;
		zMaxVrt = depth/2 - pos;
	}
	updateGL();
}

static void loadMatrix(const QMatrix4x4& m)
{
    // static to prevent glLoadMatrixf to fail on certain drivers
    static GLfloat mat[16];
    const float *data = m.constData();
    for (int index = 0; index < 16; ++index)
        mat[index] = data[index];
    glLoadMatrixf(mat);
}

void GLWidget3D::paintGL()
{
	minU = simulation->getMinU();
	maxU = simulation->getMaxU();
	minRho = simulation->getMinRho();
	maxRho = simulation->getMaxRho();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
    QMatrix4x4 projection;
    projection.ortho(-maxDim, maxDim, -maxDim, maxDim, -maxDim*(1+zoom*100), maxDim*(1+zoom*100));
    projection.translate(xPan, yPan);
    loadMatrix(projection);

	glMatrixMode(GL_MODELVIEW);
    QMatrix4x4 modelview;
    modelview.rotate(trackBall.rotation());
    modelview.scale(zoom);
    loadMatrix(modelview);

    shaderPrograms[currentShader]->bind();

    // load 1d texture
	glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, texTF);
    shaderPrograms[currentShader]->setUniformValue("COLOR", 0);

    // load texture from pbo
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, texData);
	for(int dev = 0; dev < deviceCount; dev++) {
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo[dev]);
		glTexSubImage3D(GL_TEXTURE_3D, 0, dev*width, 0, 0, width, height, depth, GL_RGBA, GL_FLOAT, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	}
	shaderPrograms[currentShader]->setUniformValue("VOLUME", 1);
	shaderPrograms[currentShader]->setUniformValue("minU", minU);
	shaderPrograms[currentShader]->setUniformValue("maxU", maxU);
	shaderPrograms[currentShader]->setUniformValue("minRho", minRho);
	shaderPrograms[currentShader]->setUniformValue("maxRho", maxRho);

	// draw domain
	glBegin(GL_QUADS);
	{
		glTexCoord3f(xMinTex, yMinTex, zMinTex); glVertex3f(xMinVrt, yMinVrt, zMaxVrt); // Bottom Left Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMinTex, zMinTex); glVertex3f(xMaxVrt, yMinVrt, zMaxVrt); // Bottom Right Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMaxTex, zMinTex); glVertex3f(xMaxVrt, yMaxVrt, zMaxVrt); // Top Right Of The Texture and Quad
		glTexCoord3f(xMinTex, yMaxTex, zMinTex); glVertex3f(xMinVrt, yMaxVrt, zMaxVrt); // Top Left Of The Texture and Quad

		// Back Face
		glTexCoord3f(xMinTex, yMinTex, zMaxTex); glVertex3f(xMinVrt, yMinVrt, zMinVrt);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(xMinTex, yMaxTex, zMaxTex); glVertex3f(xMinVrt, yMaxVrt, zMinVrt);	// Top Right Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMaxTex, zMaxTex); glVertex3f(xMaxVrt, yMaxVrt, zMinVrt);	// Top Left Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMinTex, zMaxTex); glVertex3f(xMaxVrt, yMinVrt, zMinVrt);	// Bottom Left Of The Texture and Quad

		// Top Face
		glTexCoord3f(xMinTex, yMaxTex, zMaxTex); glVertex3f(xMinVrt, yMaxVrt, zMinVrt);	// Top Left Of The Texture and Quad
		glTexCoord3f(xMinTex, yMaxTex, zMinTex); glVertex3f(xMinVrt, yMaxVrt, zMaxVrt);	// Bottom Left Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMaxTex, zMinTex); glVertex3f(xMaxVrt, yMaxVrt, zMaxVrt);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMaxTex, zMaxTex); glVertex3f(xMaxVrt, yMaxVrt, zMinVrt);	// Top Right Of The Texture and Quad

		// Bottom Face
		glTexCoord3f(xMaxTex, yMinTex, zMinTex); glVertex3f(xMaxVrt, yMinVrt, zMaxVrt);	// Bottom Left Of The Texture and Quad
		glTexCoord3f(xMinTex, yMinTex, zMinTex); glVertex3f(xMinVrt, yMinVrt, zMaxVrt);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(xMinTex, yMinTex, zMaxTex); glVertex3f(xMinVrt, yMinVrt, zMinVrt);	// Top Right Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMinTex, zMaxTex); glVertex3f(xMaxVrt, yMinVrt, zMinVrt);	// Top Left Of The Texture and Quad

		// Right face
		glTexCoord3f(xMaxTex, yMinTex, zMaxTex); glVertex3f(xMaxVrt, yMinVrt, zMinVrt);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMaxTex, zMaxTex); glVertex3f(xMaxVrt, yMaxVrt, zMinVrt);	// Top Right Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMaxTex, zMinTex); glVertex3f(xMaxVrt, yMaxVrt, zMaxVrt);	// Top Left Of The Texture and Quad
		glTexCoord3f(xMaxTex, yMinTex, zMinTex); glVertex3f(xMaxVrt, yMinVrt, zMaxVrt);	// Bottom Left Of The Texture and Quad

		// Left Face
		glTexCoord3f(xMinTex, yMinTex, zMaxTex); glVertex3f(xMinVrt, yMinVrt, zMinVrt);	// Bottom Left Of The Texture and Quad
		glTexCoord3f(xMinTex, yMinTex, zMinTex); glVertex3f(xMinVrt, yMinVrt, zMaxVrt);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(xMinTex, yMaxTex, zMinTex); glVertex3f(xMinVrt, yMaxVrt, zMaxVrt);	// Top Right Of The Texture and Quad
		glTexCoord3f(xMinTex, yMaxTex, zMaxTex); glVertex3f(xMinVrt, yMaxVrt, zMinVrt);	// Top Left Of The Texture and Quad
	}
    glEnd();

    shaderPrograms[currentShader]->release();
    glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, 0);
	glDisable(GL_TEXTURE_3D);
    glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_1D, 0);
	glDisable(GL_TEXTURE_1D);

	cursor->show();
}

void GLWidget3D::resizeGL(int vpWidth, int vpHeight)
{
	glViewport(0, 0, vpWidth, vpHeight);
}

QPointF GLWidget3D::pixelPosToViewPos(const QPointF& p)
{
    return QPointF(2.0 * float(p.x()) / width - 1.0,
                   1.0 - 2.0 * float(p.y()) / height);
}

void GLWidget3D::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    lastPos = QPoint(lastPos.x()+dx, lastPos.y()+dy);

    if (event->buttons() & Qt::LeftButton) {
        trackBall.move(pixelPosToViewPos(event->pos()), QQuaternion());
    } else {
        trackBall.release(pixelPosToViewPos(event->pos()), QQuaternion());
    }

    if (event->buttons() & Qt::RightButton) {
    	xPan += dx;
    	yPan -= dy;
    }

    if (event->buttons() & Qt::MidButton) {

    	q = trackBall.rotation().conjugate();
    	v = QVector3D(dx, -dy, 0).normalized();

    	v = q.rotatedVector(v);
    	v = v*5/zoom;

    	cursor->move(v.x(), v.y(), v.z());
    }
	updateGL();
}

void GLWidget3D::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();

    if (event->buttons() & Qt::LeftButton) {
        trackBall.push(pixelPosToViewPos(event->pos()), QQuaternion());
    }
	updateGL();
}

void GLWidget3D::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        trackBall.release(pixelPosToViewPos(event->pos()), QQuaternion());
    }
	updateGL();
}

void GLWidget3D::wheelEvent(QWheelEvent * event)
{
	zoom += (event->delta()/480.0f);
	if (zoom > 0) {
		zoomCount = 0;
	}
	if (zoom <= 0) {
		zoomCount += 1;
		zoom = -(event->delta()/480.0f)/zoomCount;
	}
	updateGL();
}
