#include "activation.h"

double Activation::step(double x)
{
    return x < 0 ? 0 : 1;
}

double Activation::sign(double x)
{
    return x < 0 ? -1 : 1;
}

double Activation::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double Activation::tangenth(double x)
{
    return tanh(x);
}

double Activation::dSigmoid(double x)
{
    return x * (1 - x);
}

arma::mat Activation::step(arma::mat X)
{
    return compute(X, STEP);
}

arma::mat Activation::sign(arma::mat X)
{
    return compute(X, SIGN);
}

arma::mat Activation::sigmoid(arma::mat X)
{
    return compute(X, SIGMOID);
}

arma::mat Activation::tangenth(arma::mat X)
{
    return compute(X, TANH);
}

arma::mat Activation::dSigmoid(arma::mat X)
{
    return compute(X, DERIVATIVE_SIGMOID);
}

arma::mat Activation::compute(arma::mat X, ActivationType type)
{
    arma::mat A(X.n_rows, X.n_cols);
    for(unsigned int i = 0 ; i < X.n_rows ; i++)
    {
        for(unsigned int j = 0 ; j < X.n_cols ; j++)
        {
            double x = X(i,j);
            A(i, j) = getValue(x, type);
        }
    }
    return A;
}

double Activation::getValue(const double x, ActivationType type)
{
    switch(type)
    {
    case STEP: return step(x);
    case SIGN: return sign(x);
    case SIGMOID: return sigmoid(x);
    case TANH: return tangenth(x);
    case DERIVATIVE_SIGMOID: return dSigmoid(x);
    }
    return x;
}
