#include "perceptron.h"

#include <iostream>

Perceptron::Perceptron(arma::mat X, arma::mat t, const int layers)
{
    this->X = X;
    this->t = t;
    this->layers = layers;
    arma::arma_rng::set_seed_random();
    Wh.randn(layers, X.n_cols);
    Wo.randn(1, layers);
}

void Perceptron::train(void)
{
    double error = 0.0;
    arma::vec e;

    do
    {
        e = arma::zeros(t.n_rows);

        for(unsigned int i = 0 ; i < X.n_rows ; i++)
        {
            arma::mat yh = Activation::sigmoid(X.row(i) * Wh.t());
            arma::mat yo = Activation::sigmoid(yh * Wo.t());
            arma::mat d = t(i) - yo;
            arma::mat eo = Activation::dSigmoid(yo) * d;
            arma::mat eh = Activation::dSigmoid(yh)%(eo*Wo);
            Wo = Wo + eo*yh*lr;
            Wh = Wh + eh.t()*X.row(i)*lr;
            e(i) = pow(eo(0), 2);
        }
        error = arma::sum(e) / X.n_rows;
        std::cout <<"error=" <<error <<std::endl;
    }while(error > minError);
}

void Perceptron::simulate(const double noise)
{
    arma::mat T = X + noise;
    arma::mat oh = Activation::sigmoid(T * Wh.t());
    arma::mat oo = Activation::sigmoid(oh * Wo.t());
    std::cout <<"Y=" <<std::endl <<oo <<std::endl;
}

void Perceptron::simulate(arma::mat X)
{
    // TODO
}
