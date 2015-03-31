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

#ifndef WG_PARAMETERS_H
#define WG_PARAMETERS_H

#include <QtWidgets>
#include "input.h"

class ParametersWidget : public QWidget
{
    Q_OBJECT

public:
    ParametersWidget(Input *in);
    QScrollArea* get();

private slots:
    void setUx(double value);
    void setUy(double value);
    void setUz(double value);
    void setRho(double value);
    void setViscosity(double value);

private:
    Input *input;

    QScrollArea *scrollArea;
	QGroupBox *groupBox;
	QGridLayout *layout;

    QLabel *deviceCountLabel;
	QLabel *widthLabel;
    QLabel *heightLabel;
    QLabel *depthLabel;
    QLabel *uxLabel;
    QLabel *uyLabel;
    QLabel *uzLabel;
    QLabel *rhoLabel;
    QLabel *viscLabel;
    QLabel *tauLabel;

    QDoubleSpinBox *uxSpinBox;
    QDoubleSpinBox *uySpinBox;
    QDoubleSpinBox *uzSpinBox;
    QDoubleSpinBox *rhoSpinBox;
    QDoubleSpinBox *viscSpinBox;

};

#endif
