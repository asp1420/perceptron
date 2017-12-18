#ifndef THRESHOLDCONSTANT_H
#define THRESHOLDCONSTANT_H

class ActivationConstants
{
    public:
        enum ActivationType
        {
            STEP,
            SIGN,
            SIGMOID,
            TANH,
            DERIVATIVE_SIGMOID
        };
};

#endif // THRESHOLDCONSTANT_H
