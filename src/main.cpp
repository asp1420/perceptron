/**
  * asp1420 - ASP
  * 22-oct-2017
  * Machine Learning (Neural Network - Backpropagation)
  *
  */

#include <QCoreApplication>

#include "perceptron.h"
#include <armadillo>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    arma::mat X;
    X
    <<1 <<1 <<1 <<1 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<1 <<1 <<1 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<0 <<0 <<0 <<1 <<arma::endr /* A word */
    <<1 <<1 <<1 <<1 <<1
    <<1 <<0 <<0 <<0 <<0
    <<1 <<1 <<1 <<1 <<1
    <<1 <<0 <<0 <<0 <<0
    <<1 <<1 <<1 <<1 <<1 <<arma::endr /* E word */
    <<0 <<1 <<1 <<1 <<0
    <<0 <<0 <<1 <<0 <<0
    <<0 <<0 <<1 <<0 <<0
    <<0 <<0 <<1 <<0 <<0
    <<0 <<1 <<1 <<1 <<0 <<arma::endr /* I word */
    <<1 <<1 <<1 <<1 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<1 <<1 <<1 <<1 <<arma::endr /* O word */
    <<1 <<0 <<0 <<0 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<1 <<1 <<1 <<1 <<arma::endr; /* U word */

    arma::mat t;
    t
    <<0.0 <<arma::endr /* 0.0 means A */
    <<0.2 <<arma::endr /* 0.2 means E */
    <<0.4 <<arma::endr /* 0.4 means I */
    <<0.6 <<arma::endr /* 0.6 means O */
    <<0.8 <<arma::endr; /* 0.8 means U */

    arma::mat N;
    N
    <<1 <<1 <<1 <<0 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<1 <<1 <<0 <<1
    <<1 <<0 <<0 <<0 <<1
    <<1 <<0 <<0 <<0 <<0 <<arma::endr; /* Noisy A word */

    Network *net;
    net = new Perceptron(X, t, 3);
    net->train();
    net->simulate();
    net->simulate(N); // Noisy input, recognized as "A" word

    return a.exec();
}
