#ifndef SUPERVISEDNETWORK_H
#define SUPERVISEDNETWORK_H

#include "network.h"
#include <armadillo>

using namespace arma;

class SupervisedNetwork : public Network {

public:
    SupervisedNetwork(mat X, mat t) : Network(X), t(t) {}
    virtual void weights() {}

protected:
    mat t;

};

#endif // SUPERVISEDNETWORK_H
