#ifndef NETWORK_H
#define NETWORK_H

#include "netconstant.h"
#include <armadillo>

class Network : public NetConstant {

public:
    Network(void) {}
    virtual void train(void) = 0;
    virtual void simulate(arma::mat X) = 0;
    virtual void simulate(const double noise = 0.0) = 0;
protected:
    arma::mat X;

};

#endif // NETWORK_H
