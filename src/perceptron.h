#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "network.h"
#include "threshold.h"
#include <armadillo>

class Perceptron : public Network
{
public:
    Perceptron(arma::mat X, arma::mat t, const int layers);

    virtual void train(void);
    virtual void simulate(arma::mat X);
    virtual void simulate(const double noise = 0.0);

private:
    arma::mat t;
    arma::mat Wh;
    arma::mat Wo;
    int layers;
};

#endif // PERCEPTRON_H
