#ifndef THRESHOLDCONSTANT_H
#define THRESHOLDCONSTANT_H

class ThresholdConstants
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
