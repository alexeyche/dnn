#pragma once

#include "io_worker.h"

#include <functional>

namespace dnn {

class Kernel {
    typedef std::function<double(double)> KernelFun;
public:
    Kernel() : fun(nullptr) {}
    virtual ~Kernel() {}
    virtual void usage() = 0;
    virtual void processSpec(string spec) = 0;

    Ptr<TimeSeries> generate(size_t dim, size_t length, double dt);
protected:
    KernelFun fun;
};

class EpspKernel : public Kernel {
public:
    void usage();
    void processSpec(string spec);
    Ptr<TimeSeries> generate(size_t dim, size_t length, double dt);
};


class KernelWorker : public IOWorker {
    typedef map<string, Ptr<Kernel> (*)()> kernels_map_type;
    template<typename INST> static Ptr<Kernel> createKernel() { return new INST; }

public:
    KernelWorker();
    ~KernelWorker();
    void usage();
    void processArgs(const vector<string> &args);
    void start(Spikework::Stack &s);
    void process(Spikework::Stack &s);

private:
    Ptr<Kernel> kernel;
    size_t kernel_length;
    double kernel_dt;
    kernels_map_type kernel_map;
};


}

