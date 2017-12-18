#include "simpleperceptron.h"

SimplePerceptron::SimplePerceptron(arma::mat X, arma::mat t)
{
    this->X = X;
    this->t = t;
    arma::arma_rng::set_seed_random();
    W = arma::randu(t.n_cols,X.n_cols);
    b = 1;
    std::cout <<W <<b;
}

void SimplePerceptron::train(void)
{
    bool pass;

    do
    {
        pass = true;
        for(unsigned int i = 0 ; i < X.n_rows ; i++)
        {
            arma::mat o = X.row(i) * W.t() + b;
            arma::mat y = Activation::step(o);
            if(y(0) != t(i))
            {
                W = W + (eta*(t.row(i) - y(0))*X.row(i));
                b = b + (eta*(t.row(i) - y(0)));
                //w(k+1) = w(k) + eta [d(k) - y(k)]x(k)
                //w(k+1) = b(k) + eta [d(k) - y(k)]
                pass = false;
            }
        }
    }while(!pass);
    std::cout <<W <<b;
}

void SimplePerceptron::simulate(const arma::mat X)
{
    arma::mat y = Activation::step(X * W.t() + b(0));
    std::cout <<"Y=" <<std::endl <<y <<std::endl;
}

void SimplePerceptron::simulate(const double noise)
{
    arma::mat T = X + noise;
    arma::mat y = Activation::step(T * W.t() + b(0));
    std::cout <<"Y=" <<std::endl <<y <<std::endl;
}
