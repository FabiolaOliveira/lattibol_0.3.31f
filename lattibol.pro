TARGET = lattibol

# Input
HEADERS += \
	src/lattibol.h \
	src/mainwindow.h \
	src/glwidget.h \
	src/glwidget_2d.h \
	src/glwidget_3d.h \
	src/input.h \
	src/output.h \
	src/simulation.h \
	src/simulation_d2q9.h \
	src/simulation_d3q19.h \
	src/streaming.h \
	src/collision.h \
	src/get_border.h \
	src/copy_border.h \
	src/apply_border.h \
	src/wg_parameters.h \
	src/wg_view.h \
	src/trackball.h \
	src/cursor.h \
	src/read_value.h
           
SOURCES += \
	src/lattibol.cpp \
	src/mainwindow.cpp \
	src/glwidget.cpp \
	src/glwidget_2d.cpp \
	src/glwidget_3d.cpp \
	src/input.cpp \
	src/output.cpp \
	src/wg_parameters.cpp \
	src/wg_view.cpp \
	src/trackball.cpp \
	src/cursor.cpp
	

CUDA_SOURCES += \
	src/simulation.cu \
	src/simulation_d2q9.cu \
	src/simulation_d3q19.cu \
	src/streaming.cu \
	src/collision.cu \
	src/get_border.cu \
	src/copy_border.cu \
	src/apply_border.cu \
	src/read_value.cu

OBJECTS_DIR = ./src/obj
MOC_DIR = ./src/moc

QT           += opengl widgets

# MPI
INCLUDEPATH += \
/usr/lib/openmpi/include \
cvmlcpp 

QMAKE_CC  	= mpicc
QMAKE_CXX  	= mpic++
QMAKE_LINK  = mpic++

QMAKE_CXXFLAGS = -std=gnu++0x

# CUDA
CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')

INCLUDEPATH += \
	$$CUDA_DIR/include \
	$(HOME)/NVIDIA_CUDA-5.5_Samples/common/inc  

QMAKE_LIBDIR += $$CUDA_DIR/lib64

LIBS += -lcudart

GENCODE_SM10    = -gencode arch=compute_10,code=sm_10
GENCODE_SM20    = -gencode arch=compute_20,code=sm_20
GENCODE_SM21    = -gencode arch=compute_20,code=sm_21
GENCODE_SM30    = -gencode arch=compute_30,code=sm_30
GENCODE_SM35    = -gencode arch=compute_35,code=sm_35  
GENCODE_FLAGS   = $$GENCODE_SM21

cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.cu.o
cuda.commands = $$CUDA_DIR/bin/nvcc -ccbin g++ $$join(INCLUDEPATH,' -I','-I') -Xcompiler -fPIC -m64 $$GENCODE_FLAGS -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_UNIX_COMPILERS += cuda
