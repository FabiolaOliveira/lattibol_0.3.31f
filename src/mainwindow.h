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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "glwidget.h"
#include "simulation.h"
#include "wg_parameters.h"
#include "wg_view.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow();

protected:
    void keyPressEvent(QKeyEvent *event);

private slots:
    void open();
    void saveVtk();
    void initTimer();
    void run();
    void pause();
    void stop();
    void about();
    void update();
    void openColormap();
    void changeShader(int index);

private:
    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void showStatusBarInfo();
    void createDockWidget();
    void createStatusBarInfo();
    void destroySimulation();
    void createShadersToolBar();

    void loadFile(QString &fileName);
    void loadColormap(QString &fileName);
    void setCurrentFile(const QString &fileName);
    QString strippedName(const QString &fullFileName);

    QString curFile;
    QMenu *fileMenu;
    QMenu *simulationMenu;
    QMenu *helpMenu;
    QToolBar *fileToolBar;
    QToolBar *simulationToolBar;
    QToolBar *viewToolBar;
    QAction *openAct;
    QAction *openColormapAct;
    QAction *saveVtkAct;
    QAction *exitAct;
    QAction *runAct;
    QAction *pauseAct;
    QAction *stopAct;
    QAction *aboutAct;
    GLWidget *glWidget;
    QTimer *timer;
	QDockWidget **dock;
	QLabel *mlupsLabel;
	QLabel *timeLabel;
	QLabel *fpsLabel;
	QLabel *stepsLabel;

	QToolBar *shadersToolBar;
	QLabel *shadersLabel;
    QComboBox *shadersComboBox;

    Simulation *sim;
    Input *input;
    Output *output;

	ParametersWidget *parameters;
	ViewWidget *view;
};

#endif
