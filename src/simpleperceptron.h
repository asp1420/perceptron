#ifndef SIMPLEPERCEPTRON_H
#define SIMPLEPERCEPTRON_H

#include "network.h"
#include "activation.h"
#include <armadillo>

class SimplePerceptron : public Network
{
public:
    SimplePerceptron(arma::mat X, arma::mat t);

    virtual void train(void);
    virtual void simulate(const arma::mat X);
    virtual void simulate(const double noise);

private:
    arma::mat t;
    arma::mat W;
    arma::vec b;
};

#endif // SIMPLEPERCEPTRON_H
