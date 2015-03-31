/***************************************************************************
 *   Copyright (C) 2015 by Fabíola Martins Campos de Oliveira and Lucas    *
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

#include <QtWidgets>
#include <mpi.h>

#include "glwidget.h"
#include "glwidget_2d.h"
#include "glwidget_3d.h"
#include "mainwindow.h"
#include "simulation.h"
#include "simulation_d2q9.h"
#include "simulation_d3q19.h"
#include "input.h"


MainWindow::MainWindow()
{
    createActions();
    createMenus();
    createToolBars();
    createStatusBar();

    initTimer();

    setWindowTitle(tr("lattibol"));
    setCurrentFile("");

    sim = NULL;
    dock = NULL;
    mlupsLabel = NULL;
    fpsLabel = NULL;
    timeLabel = NULL;
    glWidget = NULL;
    input = NULL;

    move(QPoint(200, 200));
    setMinimumSize(QSize(400, 300));
}

void MainWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape) {
    	close();
	}
    else
        QWidget::keyPressEvent(e);
}

void MainWindow::open()
{
	pause();
	QString fileName = QFileDialog::getOpenFileName(this, QString(), "./samples",
			tr("DAT file(*.dat);;Bitmap(*.bmp)"));
	if (!fileName.isEmpty()) loadFile(fileName);
}

void MainWindow::openColormap()
{
	pause();
	QString fileName = QFileDialog::getOpenFileName(this, QString(), "./colormaps",
			tr("Image file(*.png)"));
	if (!fileName.isEmpty()) loadColormap(fileName);
}

void MainWindow::saveVtk()
{
	if(curFile != "") { sim->saveVtk(); }
}

void MainWindow::initTimer()
{
	timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(update()));
}

void MainWindow::run()
{
	if(curFile != "") {
		timer->start(0);
		output->setPauseTime();
	}
}

void MainWindow::pause()
{
	timer->stop();
	if(curFile != "") output->initPauseTime();
}

void MainWindow::stop()
{
	timer->stop();
	if(curFile != "") loadFile(curFile);
}

void MainWindow::about()
{
   QMessageBox::about(this, tr("About"),
            tr("<b>lattibol</b> v0.3.28(alpha) Copyright (C) 2014  Fabíola Martins Campos de Oliveira, Lucas Monteiro Volpe <br /> <br /> This program comes with ABSOLUTELY NO WARRANTY; for details see the file COPYING. <br /> This is free software, and you are welcome to redistribute it under certain conditions; for details see the file COPYING."));
}

void MainWindow::update()
{
	sim->runLBM();
	view->refresh();
	glWidget->update();
    showStatusBarInfo();
    setStatusTip(tr("Running"));
}

void MainWindow::changeShader(int index)
{
	glWidget->setShader(index);
}


void MainWindow::createActions()
{
    openAct = new QAction(QIcon::fromTheme("document-open"), tr("&Open..."), this);
    openAct->setShortcuts(QKeySequence::Open);
    openAct->setStatusTip(tr("Open a project"));
    connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

    openColormapAct = new QAction(QIcon::fromTheme("document-open"), tr("Open &Colormap"), this);
    openColormapAct->setShortcut( QKeySequence(Qt::Key_F9));
    openColormapAct->setStatusTip(tr("Open colormap image"));
    connect(openColormapAct, SIGNAL(triggered()), this, SLOT(openColormap()));

    saveVtkAct = new QAction(QIcon::fromTheme("document-save"), tr("&Save VTK"), this);
    saveVtkAct->setShortcuts(QKeySequence::Save);
    saveVtkAct->setStatusTip(tr("Save VTK file"));
    connect(saveVtkAct, SIGNAL(triggered()), this, SLOT(saveVtk()));

    exitAct = new QAction(QIcon::fromTheme("application-exit"),tr("E&xit"), this);
    exitAct->setShortcuts(QKeySequence::Quit);
    exitAct->setStatusTip(tr("Exit the application"));
    connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

    runAct = new QAction(QIcon::fromTheme("media-playback-start"),tr("&Run"), this);
    runAct->setShortcut( QKeySequence(Qt::Key_F5));
    runAct->setStatusTip(tr("Run simulation"));
    connect(runAct, SIGNAL(triggered()), this, SLOT(run()));

    pauseAct = new QAction(QIcon::fromTheme("media-playback-pause"),tr("&Pause"), this);
    pauseAct->setShortcut( QKeySequence(Qt::Key_F6));
    pauseAct->setStatusTip(tr("Pause simulation"));
    connect(pauseAct, SIGNAL(triggered()), this, SLOT(pause()));

    stopAct = new QAction(QIcon::fromTheme("media-playback-stop"),tr("&Stop"), this);
    stopAct->setShortcut( QKeySequence(Qt::Key_F7));
    stopAct->setStatusTip(tr("Stop simulation"));
    connect(stopAct, SIGNAL(triggered()), this, SLOT(stop()));

    aboutAct = new QAction(QIcon::fromTheme("help-about"),tr("&About"), this);
    aboutAct->setStatusTip(tr("Show About box"));
    connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));
}

void MainWindow::createMenus()
{
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(openAct);
    fileMenu->addAction(openColormapAct);
    fileMenu->addAction(saveVtkAct);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAct);

    simulationMenu = menuBar()->addMenu(tr("&Simulation"));
    simulationMenu->addAction(runAct);
    simulationMenu->addAction(pauseAct);
    simulationMenu->addAction(stopAct);

    helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction(aboutAct);
}

void MainWindow::createToolBars()
{
    fileToolBar = addToolBar(tr("File"));
    fileToolBar->addAction(openAct);
    fileToolBar->addAction(openColormapAct);
    fileToolBar->addAction(saveVtkAct);

    simulationToolBar = addToolBar(tr("Simulation"));
    simulationToolBar->addAction(runAct);
    simulationToolBar->addAction(pauseAct);
    simulationToolBar->addAction(stopAct);
}

void MainWindow::createShadersToolBar()
{
    QStringList filter;
    QList<QFileInfo> files;

    shadersLabel = new QLabel(tr("Shader:"));
    shadersComboBox = new QComboBox;

    filter = QStringList("*.fsh");
    if(input->getLength() > 1) files = QDir("./shaders/3d/").entryInfoList(filter, QDir::Files | QDir::Readable);
    else files = QDir("./shaders/2d/").entryInfoList(filter, QDir::Files | QDir::Readable);
    foreach (QFileInfo file, files) {shadersComboBox->insertItem(999, file.baseName());}

    connect(shadersComboBox, SIGNAL(activated(int)),
            this, SLOT(changeShader(int)));

    shadersToolBar = addToolBar(tr("Shaders"));
    shadersToolBar->addWidget(shadersLabel);
    shadersToolBar->addWidget(shadersComboBox);
}


void MainWindow::createStatusBar()
{
    statusBar()->showMessage(tr("Ready"));
}

void MainWindow::loadFile(QString &fileName)
{
	if(!curFile.isEmpty()) destroySimulation();
    input = new Input;

    if (!input->loadParameters(fileName)) {
        QMessageBox::warning(this, tr("Error"),
                             tr("Cannot read file %1:\n") .arg(fileName));
        return;
    }
    fileName.resize(fileName.size()-3);

    if(input->getMaxDim() > 1) {
        fileName.append(tr("stl"));
    	if (!input->loadSolidsSTL(fileName)) {
    	        QMessageBox::warning(this, tr("Error"),
    	                             tr("Cannot read file %1:\n").arg(fileName));
    	        return;
    	}
    }
    else {
        fileName.append(tr("bmp"));
    	if (!input->loadSolids(fileName)) {
    		QMessageBox::warning(this, tr("Error"),
    				tr("Cannot read file %1:\n").arg(fileName));
    		return;
    	}
    }
    fileName.resize(fileName.size()-3);
    fileName.append(tr("dat"));

    setCurrentFile(fileName);
    statusBar()->showMessage(tr("File %1 loaded").arg(curFile), 0);

    if(input->getLength() > 1)	sim = new SimulationD3Q19(input);
    else sim = new SimulationD2Q9(input);
    output = sim->getOutput();

    delete glWidget;
    if(input->getLength() > 1) glWidget = new GLWidget3D(sim);
    else glWidget = new GLWidget2D(sim);

    setCentralWidget(glWidget);
    resize(glWidget->sizeHint() + sizeHint());

    createShadersToolBar();
    createDockWidget();
    createStatusBarInfo();
}

void MainWindow::loadColormap(QString &fileName)
{
    if (!glWidget->loadColormap(fileName)) {
        QMessageBox::warning(this, tr("Error"),
                             tr("Cannot read file %1:\n") .arg(fileName));
        return;
    }
}

void MainWindow::setCurrentFile(const QString &fileName)
{
    curFile = fileName;
    setWindowModified(false);

    QString shownName = curFile;
    if (curFile.isEmpty())
        shownName = "untitled.dat";
    setWindowFilePath(shownName);
}

QString MainWindow::strippedName(const QString &fullFileName)
{
    return QFileInfo(fullFileName).fileName();
}

void MainWindow::createDockWidget()
{
	parameters = new ParametersWidget(input);
	view = new ViewWidget(input, glWidget);

	dock = new QDockWidget*[2];
	dock[0] = new QDockWidget(tr("Parameters"), this);
	dock[0]->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    dock[0]->setMinimumWidth( 240 );
	dock[0]->setMaximumWidth( 300 );
    dock[0]->setWidget(parameters->get());
    addDockWidget(Qt::LeftDockWidgetArea, dock[0]);

	dock[1] = new QDockWidget(tr("View"), this);
	dock[1]->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    dock[1]->setMinimumWidth( 240 );
	dock[1]->setMaximumWidth( 300 );
    dock[1]->setWidget(view->get());
    addDockWidget(Qt::RightDockWidgetArea, dock[1]);
}

void MainWindow::createStatusBarInfo()
{
    mlupsLabel = new QLabel(tr("MLUPS"));
    timeLabel = new QLabel(tr(" "));
    fpsLabel = new QLabel(tr("fps"));
    stepsLabel = new QLabel(tr("steps"));
    statusBar()->insertPermanentWidget(0, mlupsLabel);
    statusBar()->insertPermanentWidget(1, timeLabel);
    statusBar()->insertPermanentWidget(2, fpsLabel);
    statusBar()->insertPermanentWidget(3, stepsLabel);
}

void MainWindow::showStatusBarInfo()
{
    mlupsLabel->setText(tr("%1 MLUPS").arg(output->getMLUPS(), 0 , 'f', 1));
    timeLabel->setText(tr("%1s").arg(output->getTime(), 0 , 'f', 6));
    fpsLabel->setText(tr("%1fps").arg(output->getFPS(), 0 , 'f', 1));
    stepsLabel->setText(tr("%1 steps").arg(sim->getSteps()));
    output->initFPSTime();
}

void MainWindow::destroySimulation()
{
	delete dock[0];
	delete dock[1];
	delete[] dock;
	delete mlupsLabel;
	delete timeLabel;
	delete fpsLabel;
	delete stepsLabel;
	delete input;
	delete sim;
	delete shadersToolBar;
}
