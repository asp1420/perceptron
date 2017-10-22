#include "perceptron.h"

#include <iostream>

Perceptron::Perceptron(arma::mat X, arma::mat t, const int layers)
{
    this->X = X;
    this->t = t;
    arma::arma_rng::set_seed_random();
    Wh.randu(layers, X.n_cols);
    Wo.randu(1, layers);
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
            // Forward
            arma::mat yh = Activation::sigmoid(X.row(i) * Wh.t());
            arma::mat yo = Activation::sigmoid(yh * Wo.t());
            arma::mat d = t(i) - yo;
            // Backpropagation
            arma::mat eo = Activation::dSigmoid(yo) * d;
            arma::mat eh = Activation::dSigmoid(yh)%(eo*Wo);
            // Weight update
            Wo = Wo + eo*yh*lr;
            Wh = Wh + eh.t()*X.row(i)*lr;
            e(i) = pow(eo(0), 2);
        }
        error = arma::sum(e) * 0.5; // MLS
    }while(error > minError);
    std::cout <<"error=" <<error <<std::endl;
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
    arma::mat yh = Activation::sigmoid(X * Wh.t());
    arma::mat yo = Activation::sigmoid(yh * Wo.t());
    std::cout <<"Y=" <<std::endl <<yo <<std::endl;

}
