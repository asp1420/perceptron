#ifndef SIMPLEPERCEPTRON_H
#define SIMPLEPERCEPTRON_H

#include "supervisednetwork.h"
#include "activation.h"
#include <armadillo>

using namespace arma;
using namespace std;

class LinearPerceptron : public SupervisedNetwork
{
public:
    LinearPerceptron(mat X, mat t);

    virtual void train(void);
    virtual void simulate(const mat X);
    virtual void simulate(const double noise);

    void weights();
private:
    mat W;
    vec b;
};

#endif // SIMPLEPERCEPTRON_H
