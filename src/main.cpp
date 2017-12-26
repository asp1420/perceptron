/**
  * asp1420 - ASP
  * 22-oct-2017
  * Machine Learning (Neural Network - Backpropagation)
  *
  */

#include <QCoreApplication>

#include "perceptron.h"
#include "adaline.h"
#include "simpleperceptron.h"
#include <armadillo>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    arma::mat X;
    X
    <<1 <<1 <<1 <<1 <<1 <<1 <<0 <<0 <<0 <<1 <<1 <<1 <<1 <<1 <<1 <<1 <<0 <<0 <<0 <<1 <<1 <<0 <<0 <<0 <<1 <<arma::endr /* A word */
    <<1 <<1 <<1 <<1 <<1 <<1 <<0 <<0 <<0 <<0 <<1 <<1 <<1 <<1 <<1 <<1 <<0 <<0 <<0 <<0 <<1 <<1 <<1 <<1 <<1 <<arma::endr /* E word */
    <<0 <<1 <<1 <<1 <<0 <<0 <<0 <<1 <<0 <<0 <<0 <<0 <<1 <<0 <<0 <<0 <<0 <<1 <<0 <<0 <<0 <<1 <<1 <<1 <<0 <<arma::endr /* I word */
    <<1 <<1 <<1 <<1 <<1 <<1 <<0 <<0 <<0 <<1 <<1 <<0 <<0 <<0 <<1 <<1 <<0 <<0 <<0 <<1 <<1 <<1 <<1 <<1 <<1 <<arma::endr /* O word */
    <<1 <<0 <<0 <<0 <<1 <<1 <<0 <<0 <<0 <<1 <<1 <<0 <<0 <<0 <<1 <<1 <<0 <<0 <<0 <<1 <<1 <<1 <<1 <<1 <<1 <<arma::endr; /* U word */

    arma::mat t;
    t
    <<0.0 <<arma::endr /* 0.0 means A */
    <<0.2 <<arma::endr /* 0.2 means E */
    <<0.4 <<arma::endr /* 0.4 means I */
    <<0.6 <<arma::endr /* 0.6 means O */
    <<0.8 <<arma::endr; /* 0.8 means U */

    arma::mat N1;
    N1
    <<0 <<1 <<1 <<1 <<0.7 <<1 <<0 <<0 <<0 <<1 <<1 <<1 <<0.6 <<1 <<1 <<1 <<0 <<0 <<0 <<1 <<0.4 <<0 <<0 <<0 <<0 <<arma::endr; /* Noisy A word */
    arma::mat N2;
    N2
    <<0 <<1 <<0.5 <<1 <<0 <<0 <<0 <<1 <<0 <<0 <<0 <<0 <<0.3 <<0 <<0 <<0 <<0 <<1 <<0 <<0 <<0 <<1 <<0.5 <<1 <<0 <<arma::endr; /* Noisy I word */
    arma::mat N3;
    N3
    <<0.8 <<0 <<0 <<0 <<0.7 <<1 <<0 <<0 <<0 <<1 <<1 <<0 <<0.2 <<0 <<1 <<0.8 <<0 <<0 <<0 <<0.7 <<1 <<1 <<0.8 <<1 <<1 <<arma::endr; /* Noisy U word */
    arma::mat N4;
    N4
    <<1 <<1 <<1 <<0.6 <<1 <<1 <<0 <<0 <<0 <<0 <<0.8 <<1 <<0.7 <<1 <<1 <<1 <<0 <<0 <<0 <<0 <<1 <<1 <<1 <<1 <<0.6 <<arma::endr; /* Noisy E word */


    arma::mat X2;
    X2
    <<0 <<0 <<arma::endr
    <<0 <<1 <<arma::endr
    <<1 <<0 <<arma::endr
    <<1 <<1 <<arma::endr;

    arma::mat t2;
    t2
    <<0 <<arma::endr
    <<0 <<arma::endr
    <<0 <<arma::endr
    <<1 <<arma::endr;

    arma::mat X3;
    X3
    <<-0.5 <<-0.5 <<arma::endr
    <<-0.5 <<0.5 <<arma::endr
    <<0.3 <<-0.5 <<arma::endr
    <<-0.1 <<1 <<arma::endr;

    arma::mat t3;
    t3
    <<1 <<arma::endr
    <<1 <<arma::endr
    <<0 <<arma::endr
    <<0 <<arma::endr;

    /**
     * Simple perceptron
     */
    Network *net = new SimplePerceptron(X3, t3);
    net->train();
    net->simulate();

    /**
     * Adaline
     */
    /*Network *net = new Adaline(X2, t2);
    net->train();
    net->simulate();*/

    /**
     * Multilayer perceptron
     */
    /*Network *net = new Perceptron(X, t, 4);
    net->train();
    // Test original input
    net->simulate();
    // Test noisy inputs
    net->simulate(0.02);
    // Test noisy A word
    net->simulate(N1);
    // Test noisy I word
    net->simulate(N2);
    // Test noisy U word
    net->simulate(N3);
    // Test noisy E word
    net->simulate(N4);*/

    return a.exec();
}
