QT += core
QT -= gui

CONFIG += c++11

TARGET = perceptron
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    perceptron.cpp \
    activation.cpp

LIBS += -O2 -larmadillo -llapack -lblas

HEADERS += \
    netconstant.h \
    network.h \
    perceptron.h \
    activation.h \
    activationconstant.h
