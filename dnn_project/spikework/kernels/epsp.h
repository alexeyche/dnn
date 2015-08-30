#pragma once

#include <dnn/util/func_param_parser.h>
#include <spikework/kernel.h>

namespace dnn {

struct EpspKernelOptions {
    EpspKernelOptions() : tau_rise(0.0), tau_decay(10.0), length(100), dt(1.0) {}
    double tau_rise;
    double tau_decay;

    double length;
    double dt;
};

class EpspKernel : public KernelPreprocessor<EpspKernelOptions>, public FunKernel {
public:
    void usage(ostream &str) const;
    void processSpec(const string &spec);

    Ptr<TimeSeries> operator () (Ptr<TimeSeries> x) const;
};


}



