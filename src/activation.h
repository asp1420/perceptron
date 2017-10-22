#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <armadillo>
#include <math.h>
#include "activationconstant.h"

class Activation : public ThresholdConstants
{
public:
    Activation();
    virtual ~Activation();

    static double step(double x);
    static double sign(double x);
    static double sigmoid(double x);
    static double tangenth(double x);
    static double dSigmoid(double x);
    static arma::mat step(arma::mat X);
    static arma::mat sign(arma::mat X);
    static arma::mat sigmoid(arma::mat X);
    static arma::mat tangenth(arma::mat X);
    static arma::mat dSigmoid(arma::mat X);
private:
    static double getValue(const double x, ActivationType type);
    static arma::mat compute(arma::mat X, ActivationType type);
};

inline double Activation::step(double x)
{
    return x < 0 ? 0 : 1;
}

inline double Activation::sign(double x)
{
    return x < 0 ? -1 : 1;
}

inline double Activation::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

inline double Activation::tangenth(double x)
{
    return tanh(x);
}

inline double Activation::dSigmoid(double x)
{
    return x * (1 - x);
}

inline arma::mat Activation::step(arma::mat X)
{
    return compute(X, STEP);
}

inline arma::mat Activation::sign(arma::mat X)
{
    return compute(X, SIGN);
}

inline arma::mat Activation::sigmoid(arma::mat X)
{
    return compute(X, SIGMOID);
}

inline arma::mat Activation::tangenth(arma::mat X)
{
    return compute(X, TANH);
}

inline arma::mat Activation::dSigmoid(arma::mat X)
{
    return compute(X, DERIVATIVE_SIGMOID);
}

#endif // THRESHOLD_H
