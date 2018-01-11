#ifndef ADALINE_H
#define ADALINE_H

#include "supervisednetwork.h"
#include "activation.h"
#include <armadillo>

using namespace arma;
using namespace std;

class Adaline : public SupervisedNetwork
{
public:
    Adaline(mat X, mat t);

    virtual void train(void);
    virtual void simulate(const mat X);
    virtual void simulate(const double noise);

    void weights();
private:
    mat W;
    vec b;
};

#endif // ADALINE_H
