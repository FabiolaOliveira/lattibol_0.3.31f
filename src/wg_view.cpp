/***************************************************************************
 *   Copyright (C) 2014 by Fabíola Martins Campos de Oliveira and Lucas    *
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

#include <iostream>
using namespace std;

#include "wg_view.h"
#include "input.h"
#include "glwidget.h"

QScrollArea* ViewWidget::get() { return scrollArea; }

void ViewWidget::refresh() {
	cursorXPos = input->getCursorX();
	cursorYPos = input->getCursorY();
	cursorZPos = input->getCursorZ();
	cursorXPosSlider->setValue(cursorXPos);
	cursorYPosSlider->setValue(cursorYPos);
	cursorZPosSlider->setValue(cursorZPos);
	cursorXLabel->setText(tr("Cursor X: %1").arg(cursorXPos));
	cursorYLabel->setText(tr("Cursor Y: %1").arg(cursorYPos));
	cursorZLabel->setText(tr("Cursor Z: %1").arg(cursorZPos));
    cursorUMagLabel->setText(tr("Velocity Mag: %1").arg(input->getCursorUMag()));
    cursorUXLabel->setText(tr("Velocity X: %1").arg(input->getCursorUX()));
    cursorUYLabel->setText(tr("Velocity Y: %1").arg(input->getCursorUY()));
    cursorUZLabel->setText(tr("Velocity Z: %1").arg(input->getCursorUZ()));
    cursorRhoLabel->setText(tr("Density: %1").arg(input->getCursorRho()));
}

void ViewWidget::setUpdateSteps(int value) { input->setUpdateSteps(value); }
void ViewWidget::setUMin(double value) { input->setMinU(value); }
void ViewWidget::setUMax(double value) { input->setMaxU(value); }
void ViewWidget::setRhoMin(double value) { input->setMinRho(value); }
void ViewWidget::setRhoMax(double value) { input->setMaxRho(value); }

void ViewWidget::showClipX() {
	if(clipX) glWidget->clipX(clipXPos, clipXDir);
	else glWidget->clipX(input->getWidth()-1, 0);
}
void ViewWidget::setClipX(int value) {
	if (value == 0) clipX = 0;
	else clipX = 1;
	showClipX();
}
void ViewWidget::setClipXPos(int value) {
	clipXPos = value;
    clipXCheckBox->setText(tr("Clip X: %1").arg(value));
	showClipX();
}
void ViewWidget::setClipXDir(int value) {
	if (value == 0) clipXDir = 0;
	else clipXDir = 1;
	showClipX();
}

void ViewWidget::showClipY() {
	if(clipY) glWidget->clipY(clipYPos, clipYDir);
	else glWidget->clipY(input->getHeight()-1, 0);

}
void ViewWidget::setClipY(int value) {
	if (value == 0) clipY = 0;
	else clipY = 1;
	showClipY();
}
void ViewWidget::setClipYPos(int value) {
	clipYPos = value;
    clipYCheckBox->setText(tr("Clip Y: %1").arg(value));
	showClipY();
}
void ViewWidget::setClipYDir(int value) {
	if (value == 0) clipYDir = 0;
	else clipYDir = 1;
	showClipY();
}

void ViewWidget::showClipZ() {
	if(clipZ) glWidget->clipZ(clipZPos, clipZDir);
	else glWidget->clipZ(input->getLength()-1, 0);
}
void ViewWidget::setClipZ(int value) {
	if (value == 0) clipZ = 0;
	else clipZ = 1;
	showClipZ();
}
void ViewWidget::setClipZPos(int value) {
	clipZPos = value;
    clipZCheckBox->setText(tr("Clip Z: %1").arg(value));
	showClipZ();
}
void ViewWidget::setClipZDir(int value) {
	if (value == 0) clipZDir = 0;
	else clipZDir = 1;
	showClipZ();
}

void ViewWidget::setCursorX(int value) {input->setCursorX(value);}
void ViewWidget::setCursorY(int value) {input->setCursorY(value);}
void ViewWidget::setCursorZ(int value) {input->setCursorZ(value);}


ViewWidget::ViewWidget(Input *in, GLWidget *glwg)
    : QWidget()
{

	input = in;
	glWidget = glwg;

	clipX = 0;
	clipXPos = input->getWidth()/2;
	clipXDir = 0;

	clipY = 0;
	clipYPos = input->getHeight()/2;
	clipYDir = 0;

	clipZ = 0;
	clipZPos = input->getLength()/2;
	clipZDir = 0;

	cursorXPos = clipXPos;
	cursorYPos = clipYPos;
	cursorZPos = clipZPos;

	scrollArea = new QScrollArea;
	groupBox = new QGroupBox;

    updateStepsLabel = new QLabel(tr("Update Steps:"));
    uMinLabel =  new QLabel(tr("Min U:"));
    uMaxLabel =  new QLabel(tr("Max U:"));
    rhoMinLabel =  new QLabel(tr("Min Rho:"));
    rhoMaxLabel =  new QLabel(tr("Max Rho:"));
    clipLabel =  new QLabel(tr("Clip:"));

    cursorXLabel =  new QLabel(tr("Cursor X: %1").arg(input->getCursorX()));
    cursorYLabel =  new QLabel(tr("Cursor Y: %1").arg(input->getCursorY()));
    cursorZLabel =  new QLabel(tr("Cursor Z: %1").arg(input->getCursorZ()));
    cursorUMagLabel = new QLabel(tr("Velocity Mag: %1").arg(input->getCursorUMag()));
    cursorUXLabel = new QLabel(tr("Velocity X: %1").arg(input->getCursorUX()));
    cursorUYLabel = new QLabel(tr("Velocity Y: %1").arg(input->getCursorUY()));
    cursorUZLabel = new QLabel(tr("Velocity Z: %1").arg(input->getCursorUZ()));
    cursorRhoLabel = new QLabel(tr("Density: %1").arg(input->getCursorRho()));

	cursorXPosSlider = new QSlider(Qt::Horizontal);
	cursorXPosSlider->setRange(0, input->getWidth()-1);
	cursorXPosSlider->setValue(cursorXPos);
	connect(cursorXPosSlider, SIGNAL(valueChanged(int)), this, SLOT(setCursorX(int)));

	cursorYPosSlider = new QSlider(Qt::Horizontal);
	cursorYPosSlider->setRange(0, input->getHeight()-1);
	cursorYPosSlider->setValue(cursorYPos);
	connect(cursorYPosSlider, SIGNAL(valueChanged(int)), this, SLOT(setCursorY(int)));

	cursorZPosSlider = new QSlider(Qt::Horizontal);
	cursorZPosSlider->setRange(0, input->getLength()-1);
	cursorZPosSlider->setValue(cursorZPos);
	connect(cursorZPosSlider, SIGNAL(valueChanged(int)), this, SLOT(setCursorZ(int)));

	updateStepsSpinBox = new QSpinBox();
	updateStepsSpinBox->setRange(0, 9999);
	updateStepsSpinBox->setValue(input->getUpdateSteps());
	connect(updateStepsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setUpdateSteps(int)));

    uMinSpinBox = new QDoubleSpinBox();
    uMinSpinBox->setDecimals(6);
    uMinSpinBox->setRange(0, 100);
    uMinSpinBox->setSingleStep(0.001);
    uMinSpinBox->setValue(input->getMinU());
	connect(uMinSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setUMin(double)));

    uMaxSpinBox = new QDoubleSpinBox();
    uMaxSpinBox->setDecimals(6);
    uMaxSpinBox->setRange(0, 100);
    uMaxSpinBox->setSingleStep(0.001);
    uMaxSpinBox->setValue(input->getMaxU());
	connect(uMaxSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setUMax(double)));

    rhoMinSpinBox = new QDoubleSpinBox();
    rhoMinSpinBox->setDecimals(6);
    rhoMinSpinBox->setRange(0, 999);
    rhoMinSpinBox->setSingleStep(0.001);
    rhoMinSpinBox->setValue(input->getMinRho());
	connect(rhoMinSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setRhoMin(double)));

    rhoMaxSpinBox = new QDoubleSpinBox();
    rhoMaxSpinBox->setDecimals(6);
    rhoMaxSpinBox->setRange(0, 999);
    rhoMaxSpinBox->setSingleStep(0.001);
    rhoMaxSpinBox->setValue(input->getMaxRho());
	connect(rhoMaxSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setRhoMax(double)));

    clipXCheckBox = new QCheckBox(tr("Clip X: %1").arg(input->getCursorX()));
	connect(clipXCheckBox, SIGNAL(stateChanged(int)), this, SLOT(setClipX(int)));

	clipXPosSlider = new QSlider(Qt::Horizontal);
	clipXPosSlider->setRange(1, input->getWidth()-1);
	clipXPosSlider->setValue(clipXPos);
	connect(clipXPosSlider, SIGNAL(valueChanged(int)), this, SLOT(setClipXPos(int)));

    clipXDirCheckBox = new QCheckBox(tr("Invert"));
	connect(clipXDirCheckBox, SIGNAL(stateChanged(int)), this, SLOT(setClipXDir(int)));

    clipYCheckBox = new QCheckBox(tr("Clip Y: %1").arg(input->getCursorY()));
	connect(clipYCheckBox, SIGNAL(stateChanged(int)), this, SLOT(setClipY(int)));

	clipYPosSlider = new QSlider(Qt::Horizontal);
	clipYPosSlider->setRange(1, input->getHeight()-1);
	clipYPosSlider->setValue(clipYPos);
	connect(clipYPosSlider, SIGNAL(valueChanged(int)), this, SLOT(setClipYPos(int)));

    clipYDirCheckBox = new QCheckBox(tr("Invert"));
	connect(clipYDirCheckBox, SIGNAL(stateChanged(int)), this, SLOT(setClipYDir(int)));

    clipZCheckBox = new QCheckBox(tr("Clip Z: %1").arg(input->getCursorZ()));
	connect(clipZCheckBox, SIGNAL(stateChanged(int)), this, SLOT(setClipZ(int)));

	clipZPosSlider = new QSlider(Qt::Horizontal);
	clipZPosSlider->setRange(1, input->getLength()-1);
	clipZPosSlider->setValue(clipZPos);
	connect(clipZPosSlider, SIGNAL(valueChanged(int)), this, SLOT(setClipZPos(int)));

    clipZDirCheckBox = new QCheckBox(tr("Invert"));
	connect(clipZDirCheckBox, SIGNAL(stateChanged(int)), this, SLOT(setClipZDir(int)));

	layout = new QGridLayout();
	settingsLayout = new QGridLayout();
	clipLayout = new QGridLayout();
	cursorPosLayout = new QGridLayout();
	cursorValuesLayout = new QGridLayout();

	settingsLayout->addWidget(updateStepsLabel, 10, 0);
	settingsLayout->addWidget(updateStepsSpinBox, 10, 1);
	settingsLayout->addWidget(uMinLabel, 11, 0);
	settingsLayout->addWidget(uMinSpinBox, 11, 1);
	settingsLayout->addWidget(uMaxLabel, 12, 0);
	settingsLayout->addWidget(uMaxSpinBox, 12, 1);
	settingsLayout->addWidget(rhoMinLabel, 13, 0);
	settingsLayout->addWidget(rhoMinSpinBox, 13, 1);
	settingsLayout->addWidget(rhoMaxLabel, 14, 0);
	settingsLayout->addWidget(rhoMaxSpinBox, 14, 1);
	clipLayout->addWidget(clipLabel, 15, 0);
	clipLayout->addWidget(clipXCheckBox, 16, 0);
	clipLayout->addWidget(clipXPosSlider, 16, 1);
	clipLayout->addWidget(clipXDirCheckBox, 16, 2);
	clipLayout->addWidget(clipYCheckBox, 17, 0);
	clipLayout->addWidget(clipYPosSlider, 17, 1);
	clipLayout->addWidget(clipYDirCheckBox, 17, 2);
	clipLayout->addWidget(clipZCheckBox, 18, 0);
	clipLayout->addWidget(clipZPosSlider, 18, 1);
	clipLayout->addWidget(clipZDirCheckBox, 18, 2);
	cursorPosLayout->addWidget(cursorXLabel, 19, 0);
	cursorPosLayout->addWidget(cursorXPosSlider, 19, 1);
	cursorPosLayout->addWidget(cursorYLabel, 20, 0);
	cursorPosLayout->addWidget(cursorYPosSlider, 20, 1);
	cursorPosLayout->addWidget(cursorZLabel, 21, 0);
	cursorPosLayout->addWidget(cursorZPosSlider, 21, 1);
	cursorValuesLayout->addWidget(cursorUMagLabel, 22, 0);
	cursorValuesLayout->addWidget(cursorUXLabel, 23, 0);
	cursorValuesLayout->addWidget(cursorUYLabel, 24, 0);
	cursorValuesLayout->addWidget(cursorUZLabel, 25, 0);
	cursorValuesLayout->addWidget(cursorRhoLabel, 26, 0);

	layout->setAlignment(Qt::AlignTop);
	layout->setVerticalSpacing(5);
	layout->addLayout(settingsLayout, 10, 0);
	layout->addLayout(clipLayout, 11, 0);
	layout->addLayout(cursorPosLayout, 12, 0);
	layout->addLayout(cursorValuesLayout, 13, 0);

	groupBox->setLayout(layout);
	scrollArea->setWidget(groupBox);
}
