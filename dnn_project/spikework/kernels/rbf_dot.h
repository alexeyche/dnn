#pragma once

#include <spikework/kernel.h>


namespace dnn {

struct RbfDotKernelOptions {
    RbfDotKernelOptions() : sigma(0.1) {}
    double sigma;
};


class RbfDotKernel : public Kernel<RbfDotKernelOptions> {
public:
    void usage(ostream &str) const;
    void processSpec(const string &spec);
    double operator () (const vector<double> &x, const vector<double> &y) const;
};


}