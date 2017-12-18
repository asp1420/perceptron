#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <armadillo>
#include <math.h>
#include "activationconstant.h"

class Activation : public ActivationConstants
{
public:
    Activation();

    static double step(const double x);
    static double sign(const double x);
    static double sigmoid(const double x);
    static double tangenth(const double x);
    static double dSigmoid(const double x);
    static arma::mat step(const arma::mat X);
    static arma::mat sign(const arma::mat X);
    static arma::mat sigmoid(const arma::mat X);
    static arma::mat tangenth(const arma::mat X);
    static arma::mat dSigmoid(const arma::mat X);
private:
    static double getValue(const double x, ActivationType type);
    static arma::mat compute(const arma::mat X, ActivationType type);
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

inline arma::mat Activation::step(const arma::mat X)
{
    return compute(X, STEP);
}

inline arma::mat Activation::sign(const arma::mat X)
{
    return compute(X, SIGN);
}

inline arma::mat Activation::sigmoid(const arma::mat X)
{
    return compute(X, SIGMOID);
}

inline arma::mat Activation::tangenth(const arma::mat X)
{
    return compute(X, TANH);
}

inline arma::mat Activation::dSigmoid(const arma::mat X)
{
    return compute(X, DERIVATIVE_SIGMOID);
}

inline arma::mat Activation::compute(const arma::mat X, ActivationType type)
{
    arma::mat A(X.n_rows, X.n_cols);
    for(unsigned int i = 0 ; i < X.n_rows ; i++)
    {
        for(unsigned int j = 0 ; j < X.n_cols ; j++)
        {
            double x = X(i,j);
            A(i, j) = getValue(x, type);
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
    case DERIVATIVE_SIGMOID: return dSigmoid(x);
    }
    return x;
}

#endif // THRESHOLD_H
