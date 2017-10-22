#include "activation.h"

Activation::~Activation()
{

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
