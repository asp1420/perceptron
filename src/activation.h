#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <armadillo>
#include <math.h>
#include "activationconstant.h"

class Activation : public ThresholdConstants
{
public:
    Activation();

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

#endif // THRESHOLD_H
