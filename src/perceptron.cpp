#include "perceptron.h"

#include <iostream>

Perceptron::Perceptron(mat X, mat t, const int layers) : SupervisedNetwork(X, t)
{
    arma_rng::set_seed_random();
    Wh.randu(layers, X.n_cols);
    Wo.randu(1, layers);
    cout <<"Weights initialization:" <<endl;
    weights();
}

void Perceptron::train(void)
{
    unsigned int epoch = 0;
    vec E;
    double error = 0.0;

    do
    {
        E = zeros(t.n_rows);
        for(unsigned int i = 0 ; i < X.n_rows ; i++)
        {
            // Forward
            mat yh = Activation::sigmoid(X.row(i) * Wh.t());
            mat yo = Activation::sigmoid(yh * Wo.t());
            double e = as_scalar(t(i) - yo);
            // Backpropagation
            mat eo = Activation::dSigmoid(yo) * e;
            mat eh = Activation::dSigmoid(yh) % (eo * Wo);
            // Weight update
            Wo = Wo + (eo * yh * eta);
            Wh = Wh + (eh.t() * X.row(i) * eta);
            E(i) = e * e;
        }
        error = sum(E) / X.n_rows;
        epoch++;
    }while(error > minError);

    cout <<"training statistics:" <<endl;
    cout <<"- Epoch: " <<epoch <<endl;
    cout <<"- Error: " <<error <<endl;
    cout <<endl;
}

void Perceptron::weights()
{
    cout <<"Wh=" <<Wh <<"Wo=" <<Wo <<endl;
}

void Perceptron::simulate(const double noise)
{
    cout <<"Training set (X):" <<endl;
    cout <<X <<endl;
    cout <<"Target set (t):" <<endl;
    cout <<t <<endl;
    cout <<"Weights after training:" <<endl;
    weights();
    mat T = X + noise;
    mat oh = Activation::sigmoid(T * Wh.t());
    mat oo = Activation::sigmoid(oh * Wo.t());
    cout <<"Y=" <<endl <<oo <<endl;
}

void Perceptron::simulate(const mat X)
{
    mat yh = Activation::sigmoid(X * Wh.t());
    mat yo = Activation::sigmoid(yh * Wo.t());
    cout <<"Y=" <<endl <<yo <<endl;

}
