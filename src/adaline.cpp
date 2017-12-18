#include "adaline.h"

Adaline::Adaline(arma::mat X, arma::mat t)
{
    this->X = X;
    this->t = t;
    arma::arma_rng::set_seed_random();
    W = arma::randu(t.n_cols,X.n_cols);
    b = 1;
}

void Adaline::train(void)
{
    double error = 0.0;
    arma::vec e;

    do
    {
        e = arma::zeros(t.n_rows);
        for(unsigned int i = 0 ; i < X.n_rows ; i++)
        {
            arma::mat o = Activation::sigmoid(X.row(i) * W.t() + b);
            arma::mat y = Activation::dSigmoid(o);
            arma::mat g = t(i) - o;
            W = W+(X.row(i)*eta*g(0)*y(0));
            b = b+(eta*g(0)*y);
            e(i) = pow(g(0), 2);
        }
        error = arma::sum(e);
        std::cout <<"=" <<error <<std::endl;
    }while(error > minError);
}

void Adaline::simulate(const double noise)
{
    arma::mat T = X + noise;
    arma::mat y = Activation::sigmoid(T * W.t() + b(0));
    std::cout <<"Y=" <<std::endl <<y <<std::endl;
}

void Adaline::simulate(const arma::mat X)
{
    arma::mat y = Activation::step(X * W.t() + b(0));
    std::cout <<"Y=" <<std::endl <<y <<std::endl;
}
