#include "simpleperceptron.h"

LinearPerceptron::LinearPerceptron(mat X, mat t) : SupervisedNetwork(X, t)
{
    arma_rng::set_seed_random();
    W = randu(t.n_cols,X.n_cols);
    b = 0.1;
    cout <<"Weights initialization:" <<endl;
    weights();
}

void LinearPerceptron::train(void)
{
    bool pass;
    unsigned int epoch = 0;

    do
    {
        pass = true;
        for(unsigned int i = 0 ; i < X.n_rows ; i++)
        {
            mat o = X.row(i) * W.t() + b;
            mat y = Activation::step(o);
            mat e = t.row(i) - y;
            if(t(i) != as_scalar(y))
            {
                //w(k+1) = w(k) + e(k)x(k)eta
                //w(k+1) = b(k) + e(k)eta
                W = W + (e * eta * X.row(i));
                b = b + (e * eta);
                pass = false;
            }
        }
        epoch++;

    }while(!pass && epoch < maxEpochs);

    cout <<"training statistics:" <<endl;
    cout <<"- Epoch: " <<epoch <<endl;
    cout <<endl;
}

void LinearPerceptron::weights()
{
    cout <<"W=" <<W <<"b=" <<b <<endl;
}

void LinearPerceptron::simulate(const mat X)
{
    mat y = Activation::step(X * W.t() + as_scalar(b));
    cout <<"Y=" <<endl <<y <<endl;
}

void LinearPerceptron::simulate(const double noise)
{
    cout <<"Training set (X):" <<endl;
    cout <<X <<endl;
    cout <<"Target set (t):" <<endl;
    cout <<t <<endl;
    cout <<"Weights after training:" <<endl;
    weights();
    mat T = X + noise;
    mat y = Activation::step(T * W.t() + as_scalar(b));
    cout <<"Output:" <<endl;
    cout <<y <<endl;
}
