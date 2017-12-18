#ifndef ADALINE_H
#define ADALINE_H

#include "network.h"
#include "activation.h"
#include <armadillo>

class Adaline : public Network
{
public:
    Adaline(arma::mat X, arma::mat t);

    virtual void train(void);
    virtual void simulate(const arma::mat X);
    virtual void simulate(const double noise);

private:
    arma::mat t;
    arma::mat W;
    arma::vec b;
};

#endif // ADALINE_H
