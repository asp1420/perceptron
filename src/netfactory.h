#ifndef NETFACTORY_H
#define NETFACTORY_H

#include "network.h"
#include "netconstant.h"
#include "perceptron"
#include "adaline.h"
#include <armadillo>

class NetFactory : public NetConstant
{
public:
    static NetFactory* instance = nullptr;
    static NetFactory* instance()
    {
        if(!instance)
        {
            instance = new NetFactory();
        }
        return instance;
    }

    Network* make(NetType type, arma::mat X, arma::vec t, double l = 1.0)
    {
        Network *net = nullptr;
        switch (type) {
        case PERCEPTRON:
            // TODO
            break;
        case ADALINE:
            net = new Adaline(X, t);
            break;
        case MULTILAYER:
            net = new Perceptron(X, t, l);
            break;
        default:
            break;
        }
    }

private:
    NetFactory() {}
};

#endif // NETFACTORY_H
