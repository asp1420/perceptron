#include "adaline.h"

Adaline::Adaline(mat X, mat t) : SupervisedNetwork(X, t)
{
    arma_rng::set_seed_random();
    W = randu(t.n_cols,X.n_cols);
    b = 0.1;
    cout <<"Weights initialization:" <<endl;
    weights();
}

void Adaline::train(void)
{
    unsigned int epoch = 0;
    vec E;
    double error = 0.0;

    do
    {
        E = zeros(t.n_rows);
        for(unsigned int i = 0 ; i < X.n_rows ; i++)
        {
            mat v = Activation::sigmoid(X.row(i) * W.t() + b);
            mat y = Activation::dSigmoid(v);
            mat e = t(i) - v;
            W = W + (e * y * eta * X.row(i));
            b = b + (e * y * eta);
            E(i) = as_scalar(e * e);
        }
        error = sum(E) / X.n_rows;
        cout <<"error=" <<error <<endl;
        epoch++;
    }while(error > minError && epoch < maxEpochs);

    cout <<"training statistics:" <<endl;
    cout <<"- Epoch: " <<epoch <<endl;
    cout <<"- Error: " <<error <<endl;
    cout <<endl;
}

void Adaline::weights()
{
    cout <<"W=" <<W <<"b=" <<b <<endl;
}

void Adaline::simulate(const double noise)
{
    cout <<"Training set (X):" <<endl;
    cout <<X <<endl;
    cout <<"Target set (t):" <<endl;
    cout <<t <<endl;
    cout <<"Weights after training:" <<endl;
    weights();
    mat T = X + noise;
    mat y = Activation::sigmoid(T * W.t() + as_scalar(b));
    cout <<"Output:" <<endl;
    cout <<y <<endl;
}

void Adaline::simulate(const mat X)
{
    mat y = Activation::step(X * W.t() + as_scalar(b));
    cout <<"Y=" <<endl <<y <<endl;
}
