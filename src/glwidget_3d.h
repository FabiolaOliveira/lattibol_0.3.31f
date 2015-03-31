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

#ifndef GLWIDGET_3D_H
#define GLWIDGET_3D_H

#include <QGLWidget>
#include <QtOpenGL>
#include "glwidget.h"
#include "simulation.h"
#include "trackball.h"
#include "cursor.h"

class GLWidget3D: public GLWidget
{
    Q_OBJECT

public:
    GLWidget3D(Simulation *sim = 0);
    ~GLWidget3D();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    void clipX(int pos, bool dir);
    void clipY(int pos, bool dir);
    void clipZ(int pos, bool dir);

    // public slots:

    //signals:

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void showAll();

    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void wheelEvent(QWheelEvent * event);

private:
    QPointF pixelPosToViewPos(const QPointF& p);

    int width;
    int height;
    int depth;
    int maxDim;
    int deviceCount;
    Simulation *simulation;

	float xMinTex;
	float xMaxTex;
	float yMinTex;
	float yMaxTex;
	float zMinTex;
	float zMaxTex;

	float xMinVrt;
	float xMaxVrt;
	float yMinVrt;
	float yMaxVrt;
	float zMinVrt;
	float zMaxVrt;

    TrackBall trackBall;
    QQuaternion q;
    QVector3D v;

    QPoint lastPos;

    float zoom;
    int zoomCount;

    float xPan, yPan;
};

#endif
