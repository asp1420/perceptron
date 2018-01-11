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
#include "supervisednetwork.h"
#include <armadillo>

using namespace arma;
using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    mat P;
    P.load("multi.csv", csv_ascii);
    mat X3 = P(span(0,P.n_rows-1), span(0,P.n_cols-2));
    mat t3 = P(span(0,P.n_rows-1), span(P.n_cols-1,P.n_cols-1));

    /**
     * Base ANN
     */
    SupervisedNetwork *net;

    /**
     * Simple perceptron
     */
    net = new LinearPerceptron(X3, t3);

    /**
     * ADALINE
     */
    //net = new Adaline(X3, t3);

    /**
     * Multilayer perceptron
     */
    //net = new Perceptron(X3, t3, 8);

    net->train();
    net->simulate();

    return a.exec();
}
