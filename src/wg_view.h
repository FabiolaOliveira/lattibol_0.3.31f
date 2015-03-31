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

#ifndef WG_VIEW_H
#define WG_VIEW_H

#include <QtWidgets>
#include "input.h"
#include "glwidget.h"

class ViewWidget : public QWidget
{
    Q_OBJECT

public:
    ViewWidget(Input *in, GLWidget *glwg);
    QScrollArea* get();
    void refresh();

private slots:
    void setUpdateSteps(int value);
    void setUMin(double value);
    void setUMax(double value);
    void setRhoMin(double value);
    void setRhoMax(double value);

    void showClipX();
    void setClipX(int value);
    void setClipXPos(int value);
    void setClipXDir(int value);

    void showClipY();
    void setClipY(int value);
    void setClipYPos(int value);
    void setClipYDir(int value);

    void showClipZ();
    void setClipZ(int value);
    void setClipZPos(int value);
    void setClipZDir(int value);

    void setCursorX(int value);
    void setCursorY(int value);
    void setCursorZ(int value);

private:
    Input *input;
    GLWidget *glWidget;

    QScrollArea *scrollArea;
	QGroupBox *groupBox;
	QGridLayout *layout;
	QGridLayout *settingsLayout;
	QGridLayout *clipLayout;
	QGridLayout *cursorPosLayout;
	QGridLayout *cursorValuesLayout;

    QLabel *updateStepsLabel;
    QLabel *uMinLabel;
    QLabel *uMaxLabel;
    QLabel *rhoMinLabel;
    QLabel *rhoMaxLabel;
    QLabel *clipLabel;

    QSpinBox *updateStepsSpinBox;
    QDoubleSpinBox *uMinSpinBox;
    QDoubleSpinBox *uMaxSpinBox;
    QDoubleSpinBox *rhoMinSpinBox;
    QDoubleSpinBox *rhoMaxSpinBox;
    QCheckBox *clipXCheckBox;
    QSlider *clipXPosSlider;
    QCheckBox *clipXDirCheckBox;
    QCheckBox *clipYCheckBox;
    QSlider *clipYPosSlider;
    QCheckBox *clipYDirCheckBox;
    QCheckBox *clipZCheckBox;
    QSlider *clipZPosSlider;
    QCheckBox *clipZDirCheckBox;

    QLabel *cursorXLabel;
    QLabel *cursorYLabel;
    QLabel *cursorZLabel;
    QLabel *cursorUMagLabel;
    QLabel *cursorUXLabel;
    QLabel *cursorUYLabel;
    QLabel *cursorUZLabel;
    QLabel *cursorRhoLabel;
    QSlider *cursorXPosSlider;
    QSlider *cursorYPosSlider;
    QSlider *cursorZPosSlider;


    bool clipX;
    int clipXPos;
    bool clipXDir;
    bool clipY;
    int clipYPos;
    bool clipYDir;
    bool clipZ;
    int clipZPos;
    bool clipZDir;

    int cursorXPos;
    int cursorYPos;
    int cursorZPos;
};

#endif
