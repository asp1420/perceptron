#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "supervisednetwork.h"
#include "activation.h"
#include <armadillo>

using namespace arma;
using namespace std;

class Perceptron : public SupervisedNetwork
{
public:
    Perceptron(mat X, mat t, const int layers);

    virtual void train(void);
    virtual void simulate(const mat X);
    virtual void simulate(const double noise);

    void weights();
private:
    mat Wh;
    mat Wo;
};

#endif // PERCEPTRON_H
