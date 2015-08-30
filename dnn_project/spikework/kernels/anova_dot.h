#pragma once

#include <spikework/kernel.h>


namespace dnn {

struct AnovaDotKernelOptions {
    AnovaDotKernelOptions() : sigma(0.1), power(1.0) {}
    double sigma;
    double power;
};


class AnovaDotKernel : public Kernel<AnovaDotKernelOptions> {
public:
    void usage(ostream &str) const;
    void processSpec(const string &spec);
    double operator () (const vector<double> &x, const vector<double> &y) const;
};


}