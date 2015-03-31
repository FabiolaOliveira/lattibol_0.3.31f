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

#include "wg_parameters.h"
#include "input.h"

QScrollArea* ParametersWidget::get() { return scrollArea; }

void ParametersWidget::setUx(double value) { input->setUx(value); }
void ParametersWidget::setUy(double value) { input->setUy(value); }
void ParametersWidget::setUz(double value) { input->setUz(value); }
void ParametersWidget::setRho(double value) { input->setRho(value); }
void ParametersWidget::setViscosity(double value)
{
	input->setViscosity(value);
	tauLabel->setText(tr("Relaxation: %1").arg(3.*value + 0.5));
}

ParametersWidget::ParametersWidget(Input *in)
    : QWidget()
{

	input = in;

	scrollArea = new QScrollArea;
	groupBox = new QGroupBox;

    deviceCountLabel = new QLabel(tr("Devices: %1").arg(input->getDeviceCount()));
    widthLabel = new QLabel(tr("Width: %1").arg(input->getWidth()));
    heightLabel = new QLabel(tr("Height: %1").arg(input->getHeight()));
    depthLabel = new QLabel(tr("Depth: %1").arg(input->getLength()));
    uxLabel = new QLabel(tr("Inlet Ux:"));
    uyLabel = new QLabel(tr("Inlet Uy:"));
    uzLabel = new QLabel(tr("Inlet Uz:"));
    rhoLabel = new QLabel(tr("Outlet Rho:"));
    viscLabel = new QLabel(tr("Viscosity:"));
    tauLabel = new QLabel(tr("Relaxation: %1").arg(3.*input->getViscosity() + 0.5));

    uxSpinBox = new QDoubleSpinBox();
    uxSpinBox->setDecimals(6);
    uxSpinBox->setRange(0, 0.2);
    uxSpinBox->setSingleStep(0.001);
    uxSpinBox->setValue(input->getUx());
	connect(uxSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setUx(double)));

    uySpinBox = new QDoubleSpinBox();
    uySpinBox->setDecimals(6);
    uySpinBox->setRange(0, 0.2);
    uySpinBox->setSingleStep(0.001);
    uySpinBox->setValue(input->getUy());
	connect(uySpinBox, SIGNAL(valueChanged(double)), this, SLOT(setUy(double)));

    uzSpinBox = new QDoubleSpinBox();
    uzSpinBox->setDecimals(6);
    uzSpinBox->setRange(0, 0.2);
    uzSpinBox->setSingleStep(0.001);
    uzSpinBox->setValue(input->getUz());
	connect(uzSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setUz(double)));

    rhoSpinBox = new QDoubleSpinBox();
    rhoSpinBox->setDecimals(6);
    rhoSpinBox->setRange(0, 200000);
    rhoSpinBox->setSingleStep(0.01);
    rhoSpinBox->setValue(input->getRho());
	connect(rhoSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setRho(double)));

    viscSpinBox = new QDoubleSpinBox();
    viscSpinBox->setDecimals(6);
    viscSpinBox->setRange(0, 999);
    viscSpinBox->setSingleStep(0.001);
    viscSpinBox->setValue(input->getViscosity());
	connect(viscSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setViscosity(double)));

    layout = new QGridLayout();
    layout->setAlignment(Qt::AlignTop);
    layout->setVerticalSpacing(5);
    layout->addWidget(deviceCountLabel, 0, 0);
    layout->addWidget(widthLabel, 1, 0);
	layout->addWidget(heightLabel, 2, 0);
	layout->addWidget(depthLabel, 3, 0);
	layout->addWidget(uxLabel, 4, 0);
	layout->addWidget(uxSpinBox, 4, 1);
	layout->addWidget(uyLabel, 5, 0);
	layout->addWidget(uySpinBox, 5, 1);
	layout->addWidget(uzLabel, 6, 0);
	layout->addWidget(uzSpinBox, 6, 1);
	layout->addWidget(rhoLabel, 7, 0);
	layout->addWidget(rhoSpinBox, 7, 1);
	layout->addWidget(viscLabel, 8, 0);
	layout->addWidget(viscSpinBox, 8, 1);
	layout->addWidget(tauLabel, 9, 0);

	groupBox->setLayout(layout);
	scrollArea->setWidget(groupBox);
}
