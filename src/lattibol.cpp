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

#include <QApplication>
#include <QDesktopWidget>

#include <cuda_runtime.h>
#include <mpi.h>

#include "lattibol.h"
#include "input.h"
#include "output.h"
#include "simulation.h"

#include "mainwindow.h"

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

    QApplication app(argc, argv);
    app.setOrganizationName("Unicamp");
    app.setApplicationName("lattibol");
    MainWindow mainWin;
    mainWin.resize(mainWin.sizeHint());
    mainWin.show();

    return app.exec();
}
