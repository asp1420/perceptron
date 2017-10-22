#include <QCoreApplication>

#include "perceptron.h"
#include <armadillo>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    arma::mat X;
    X
    <<1 <<0 <<0 <<0 <<0 <<0 <<0 <<0 <<0 <<0 <<0 <<1<<arma::endr
    <<1 <<1 <<0 <<0 <<0 <<0 <<0 <<0 <<0 <<0 <<1 <<1<<arma::endr
    <<1 <<1 <<1 <<0 <<0 <<0 <<0 <<0 <<0 <<1 <<1 <<1<<arma::endr
    <<1 <<1 <<1 <<1 <<0 <<0 <<0 <<0 <<1 <<1 <<1 <<1<<arma::endr
    <<1 <<1 <<1 <<1 <<1 <<0 <<0 <<0 <<0 <<0 <<1 <<1<<arma::endr
    <<1 <<1 <<0 <<0 <<0 <<0 <<1 <<1 <<0 <<0 <<1 <<0<<arma::endr
    <<0 <<0 <<0 <<0 <<0 <<0 <<1 <<1 <<1 <<1 <<1 <<1<<arma::endr
    <<1 <<1 <<1 <<1 <<1 <<1 <<1 <<0 <<0 <<0 <<0 <<0<<arma::endr;

    arma::mat t;
    t
    <<0.1<<arma::endr
    <<0.2<<arma::endr
    <<0.3<<arma::endr
    <<0.4<<arma::endr
    <<0.5<<arma::endr
    <<0.6<<arma::endr
    <<0.7<<arma::endr
    <<0.8<<arma::endr;

    Network *net;
    net = new Perceptron(X, t, 3);
    net->train();
    net->simulate();

    return a.exec();
}
