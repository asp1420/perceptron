#ifndef NETWORK_H
#define NETWORK_H

#include "netconstant.h"
#include <armadillo>

using namespace arma;

class Network : public NetConstant {

public:
    Network(mat X) : X(X) {}
    virtual void train(void) = 0;
    virtual void simulate(const mat X) = 0;
    virtual void simulate(const double noise = 0.0) = 0;

protected:
    mat X;

};

#endif // NETWORK_H
