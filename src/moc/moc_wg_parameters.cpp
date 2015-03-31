/****************************************************************************
** Meta object code from reading C++ file 'wg_parameters.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.1.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../wg_parameters.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'wg_parameters.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.1.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_ParametersWidget_t {
    QByteArrayData data[8];
    char stringdata[63];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_ParametersWidget_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_ParametersWidget_t qt_meta_stringdata_ParametersWidget = {
    {
QT_MOC_LITERAL(0, 0, 16),
QT_MOC_LITERAL(1, 17, 5),
QT_MOC_LITERAL(2, 23, 0),
QT_MOC_LITERAL(3, 24, 5),
QT_MOC_LITERAL(4, 30, 5),
QT_MOC_LITERAL(5, 36, 5),
QT_MOC_LITERAL(6, 42, 6),
QT_MOC_LITERAL(7, 49, 12)
    },
    "ParametersWidget\0setUx\0\0value\0setUy\0"
    "setUz\0setRho\0setViscosity\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ParametersWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x08,
       4,    1,   42,    2, 0x08,
       5,    1,   45,    2, 0x08,
       6,    1,   48,    2, 0x08,
       7,    1,   51,    2, 0x08,

 // slots: parameters
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    3,

       0        // eod
};

void ParametersWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ParametersWidget *_t = static_cast<ParametersWidget *>(_o);
        switch (_id) {
        case 0: _t->setUx((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->setUy((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->setUz((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->setRho((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->setViscosity((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject ParametersWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_ParametersWidget.data,
      qt_meta_data_ParametersWidget,  qt_static_metacall, 0, 0}
};


const QMetaObject *ParametersWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ParametersWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ParametersWidget.stringdata))
        return static_cast<void*>(const_cast< ParametersWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int ParametersWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
