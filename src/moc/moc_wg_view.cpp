/****************************************************************************
** Meta object code from reading C++ file 'wg_view.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.1.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../wg_view.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'wg_view.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.1.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_ViewWidget_t {
    QByteArrayData data[23];
    char stringdata[232];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_ViewWidget_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_ViewWidget_t qt_meta_stringdata_ViewWidget = {
    {
QT_MOC_LITERAL(0, 0, 10),
QT_MOC_LITERAL(1, 11, 14),
QT_MOC_LITERAL(2, 26, 0),
QT_MOC_LITERAL(3, 27, 5),
QT_MOC_LITERAL(4, 33, 7),
QT_MOC_LITERAL(5, 41, 7),
QT_MOC_LITERAL(6, 49, 9),
QT_MOC_LITERAL(7, 59, 9),
QT_MOC_LITERAL(8, 69, 9),
QT_MOC_LITERAL(9, 79, 8),
QT_MOC_LITERAL(10, 88, 11),
QT_MOC_LITERAL(11, 100, 11),
QT_MOC_LITERAL(12, 112, 9),
QT_MOC_LITERAL(13, 122, 8),
QT_MOC_LITERAL(14, 131, 11),
QT_MOC_LITERAL(15, 143, 11),
QT_MOC_LITERAL(16, 155, 9),
QT_MOC_LITERAL(17, 165, 8),
QT_MOC_LITERAL(18, 174, 11),
QT_MOC_LITERAL(19, 186, 11),
QT_MOC_LITERAL(20, 198, 10),
QT_MOC_LITERAL(21, 209, 10),
QT_MOC_LITERAL(22, 220, 10)
    },
    "ViewWidget\0setUpdateSteps\0\0value\0"
    "setUMin\0setUMax\0setRhoMin\0setRhoMax\0"
    "showClipX\0setClipX\0setClipXPos\0"
    "setClipXDir\0showClipY\0setClipY\0"
    "setClipYPos\0setClipYDir\0showClipZ\0"
    "setClipZ\0setClipZPos\0setClipZDir\0"
    "setCursorX\0setCursorY\0setCursorZ\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ViewWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      20,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,  114,    2, 0x08,
       4,    1,  117,    2, 0x08,
       5,    1,  120,    2, 0x08,
       6,    1,  123,    2, 0x08,
       7,    1,  126,    2, 0x08,
       8,    0,  129,    2, 0x08,
       9,    1,  130,    2, 0x08,
      10,    1,  133,    2, 0x08,
      11,    1,  136,    2, 0x08,
      12,    0,  139,    2, 0x08,
      13,    1,  140,    2, 0x08,
      14,    1,  143,    2, 0x08,
      15,    1,  146,    2, 0x08,
      16,    0,  149,    2, 0x08,
      17,    1,  150,    2, 0x08,
      18,    1,  153,    2, 0x08,
      19,    1,  156,    2, 0x08,
      20,    1,  159,    2, 0x08,
      21,    1,  162,    2, 0x08,
      22,    1,  165,    2, 0x08,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void, QMetaType::Double,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,

       0        // eod
};

void ViewWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ViewWidget *_t = static_cast<ViewWidget *>(_o);
        switch (_id) {
        case 0: _t->setUpdateSteps((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->setUMin((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 2: _t->setUMax((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->setRhoMin((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->setRhoMax((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->showClipX(); break;
        case 6: _t->setClipX((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->setClipXPos((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->setClipXDir((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->showClipY(); break;
        case 10: _t->setClipY((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: _t->setClipYPos((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 12: _t->setClipYDir((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 13: _t->showClipZ(); break;
        case 14: _t->setClipZ((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 15: _t->setClipZPos((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 16: _t->setClipZDir((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 17: _t->setCursorX((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 18: _t->setCursorY((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 19: _t->setCursorZ((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject ViewWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_ViewWidget.data,
      qt_meta_data_ViewWidget,  qt_static_metacall, 0, 0}
};


const QMetaObject *ViewWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ViewWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ViewWidget.stringdata))
        return static_cast<void*>(const_cast< ViewWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int ViewWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 20)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 20;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 20)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 20;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
