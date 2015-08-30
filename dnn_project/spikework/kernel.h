#pragma once

#include <unordered_map>
#include <functional>

using std::unordered_map;

#include <dnn/util/time_series.h>
#include <dnn/util/log/log.h>

#include "io_worker.h"
#include "conv.h"

namespace dnn {


class KernelBase {
public:
    virtual void usage(ostream &o) const = 0;
    virtual ~KernelBase() {}
    virtual void processSpec(const string &s) {}
};

class IKernel : public KernelBase {
public:
    virtual double operator () (const vector<double> &x, const vector<double> &y) const = 0;

    double process(Ptr<TimeSeries> x, Ptr<TimeSeries> y) const {
        assert((x->dim() == y->dim()) && (x->length() == y->length()));
        double integral = 0.0;
        for(size_t i=0; i<x->length(); ++i) {
            vector<double> xc = x->getColumnVector(i);
            vector<double> yc = y->getColumnVector(i);
            const IKernel &self(*this);
            integral += self(xc, yc);
        }
        return integral/x->length();
    }
};

class IKernelPreprocessor : public KernelBase {
public:
    virtual Ptr<TimeSeries> operator () (Ptr<TimeSeries> x) const = 0;
};




struct EmptyOptions {};

template <typename Options = EmptyOptions>
class Kernel : public IKernel {
protected:
    Options o;
};

template <typename Options = EmptyOptions>
class KernelPreprocessor : public IKernelPreprocessor {
protected:
    Options o;
};

class FunKernel {
    typedef std::function<double(double)> KernelFun;
public:
    Ptr<TimeSeries> generate(size_t dim, size_t length, double dt) const {
        if(fun == nullptr) {
            throw dnnException() << "Need to specify kernel function before generating\n";
        }
        Ptr<TimeSeries> out(Factory::inst().createObject<TimeSeries>());
        for(size_t di=0; di<dim; ++di) {
            double max_t = length * dt;
            for(double s=0; s<max_t; s+=dt) {
                double v = fun(s);
                out->addValue(di, v);
            }
        }
        return out;
    }
    Ptr<TimeSeries> convolve(Ptr<TimeSeries> x, double length, double dt) const {
        Spikework::Stack s;

        s.push(x);
        s.push(generate(1, length, dt));

        ConvWorker conv;
        conv.process(s);
        return s.pop().as<TimeSeries>();
    }
protected:
    KernelFun fun;
};


class KernelWorker : public IOWorker {

public:
    ~KernelWorker();
    void usage();
    void processArgs(vector<string> &args);
    void start(Spikework::Stack &s);
    void process(Spikework::Stack &s);



private:
    Ptr<IKernel> kernel;
    Ptr<IKernelPreprocessor> kernel_proc;


    string text_file;
};



}
