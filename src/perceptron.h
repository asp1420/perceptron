#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "network.h"
#include "activation.h"
#include <armadillo>

class Perceptron : public Network
{
public:
    Perceptron(arma::mat X, arma::mat t, const int layers);

    virtual void train(void);
    virtual void simulate(const arma::mat X);
    virtual void simulate(const double noise);

private:
    arma::mat t;
    arma::mat Wh;
    arma::mat Wo;
};

#endif // PERCEPTRON_H
