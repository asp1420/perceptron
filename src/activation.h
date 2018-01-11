#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <armadillo>
#include <math.h>
#include "activationconstant.h"

using namespace arma;
using namespace std;

class Activation : public ActivationConstants
{
public:
    Activation(){}

    static double step(const double x);
    static double sign(const double x);
    static double sigmoid(const double x);
    static double tangenth(const double x);
    static double dSigmoid(const double x);
    static mat step(const mat X);
    static mat sign(const mat X);
    static mat sigmoid(const mat X);
    static mat tangenth(const mat X);
    static mat dSigmoid(const mat X);
private:
    static double getValue(const double x, ActivationType type);
    static mat compute(const mat X, ActivationType type);
};

inline double Activation::step(const double x)
{
    return x < 0 ? 0 : 1;
}

inline double Activation::sign(const double x)
{
    return x < 0 ? -1 : 1;
}

inline double Activation::sigmoid(const double x)
{
    return 1 / (1 + exp(-x));
}

inline double Activation::tangenth(const double x)
{
    return tanh(x);
}

inline double Activation::dSigmoid(const double x)
{
    return x * (1 - x);
}

inline mat Activation::step(const mat X)
{
    return compute(X, STEP);
}

inline mat Activation::sign(const mat X)
{
    return compute(X, SIGN);
}

inline mat Activation::sigmoid(const mat X)
{
    return compute(X, SIGMOID);
}

inline mat Activation::tangenth(const mat X)
{
    return compute(X, TANH);
}

inline mat Activation::dSigmoid(const mat X)
{
    return compute(X, DSIGMOID);
}

inline mat Activation::compute(const mat X, ActivationType type)
{
    mat A(X.n_rows, X.n_cols);
    for(unsigned int i = 0 ; i < X.n_rows ; i++)
    {
        for(unsigned int j = 0 ; j < X.n_cols ; j++)
        {
            A(i, j) = getValue(X(i,j), type);
        }
    }
    return A;
}

inline double Activation::getValue(const double x, ActivationType type)
{
    switch(type)
    {
    case STEP: return step(x);
    case SIGN: return sign(x);
    case SIGMOID: return sigmoid(x);
    case TANH: return tangenth(x);
    case DSIGMOID: return dSigmoid(x);
    }
    return x;
}

#endif // THRESHOLD_H
