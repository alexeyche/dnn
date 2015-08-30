#pragma once

#include <spikework/kernel.h>


namespace dnn {

class IKernel;

struct ShoeKernelOptions {
    ShoeKernelOptions() : sigma(0.1) {}
    string kernel_spec;
    double sigma;
};


class ShoeKernel : public Kernel<ShoeKernelOptions> {
public:
    ~ShoeKernel() {
        if(kernel) {
            delete kernel.ptr();
        }
    }
    void usage(ostream &str) const;
    void processSpec(const string &spec);
    double operator () (const vector<double> &x, const vector<double> &y) const;
private:
    Ptr<IKernel> kernel;
};


}