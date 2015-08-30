
#include "kernel_old.h"
#include "conv.h"
#include "fft.h"

#include <dnn/util/time_series.cpp>
#include <dnn/util/util.h>
#include <dnn/util/option_parser.h>


namespace dnn {

// generic kernel creation
Ptr<TimeSeries> Kernel::generate(size_t dim, size_t length, double dt) {
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

// kernels
void EpspKernel::usage() {
    cout << "kernel that specified by exponential rise and decay, or decay only: Epsp(5, 15), Epsp(15), accordingly\n";
}
void EpspKernel::processSpec(string spec) {
    vector<string> spec_spl = split(spec, ',');
    if(spec_spl.size() == 1) {
        double tau_decay = std::stof(trimC(spec_spl[0], " \t()"));
        fun = [=](double t) {
            return exp(-t/tau_decay);
        };
    } else
    if(spec_spl.size() == 2) {
        double tau_rise = std::stof(trimC(spec_spl[0], " \t()"));
        double tau_decay = std::stof(trimC(spec_spl[1], " \t()"));
        if(fabs(tau_rise - tau_decay) < 1e-06) {
            tau_decay += 1e-05;
        }
        fun = [=](double t) {
            return (1/(1-tau_rise/tau_decay)) * (exp(-t/tau_decay) - exp(-t/tau_rise));
        };
    } else {
        stringstream ss;
        for(auto &v: spec_spl) {
            ss << v;
        }
        throw dnnException() << "EpspKernel: Unexpected parameters: " << ss.str() << "\n";
    }
}



// kernel processor

KernelWorker::KernelWorker() : kernel_length(100), kernel_dt(1.0) {
    kernel_map["Epsp"] = &createKernel<EpspKernel>;
}

KernelWorker::~KernelWorker() {
    if(kernel) {
        delete kernel.ptr();
    }
}

void KernelWorker::usage() {
    cout << "KernelWorker applying specifed kernel\n";
    cout << "\t--kernel,  -k  kernel specification (required)\n";
    cout << "\t--length,  -l  kernel length (optional, default " << kernel_length << ")\n";
    cout << "\t--dt,      -d  kernel resolution (optional, default " << kernel_dt << ")\n";
    cout << "\n";
    cout << "available kernels:\n";
    for(const auto &k: kernel_map) {
        cout << "\t" <<  k.first << "\t";
        k.second()->usage();
    }
    cout << "\n";
    IOWorker::usage();
}

void KernelWorker::processArgs(const vector<string> &args) {
    IOWorker::processArgs(args);
    OptionParser op(args);
    string kernel_spec;
    op.option("--kernel", "-k", kernel_spec, true);
    op.option("--length", "-l", kernel_length, false);
    op.option("--dt", "-d", kernel_dt, false);
    for(const auto &k: kernel_map) {
        if(strStartsWith(trimC(kernel_spec), k.first)) {
            kernel = k.second();
            replaceStr(kernel_spec, k.first, "", 1);

            kernel->processSpec(kernel_spec);
        }
    }
    if(!kernel) {
        throw dnnException() << "Can't recognize kernel specification: " << kernel_spec << "\n";
    }
}

void KernelWorker::start(Spikework::Stack &s) {
    IOWorker::start(s);
}

void KernelWorker::process(Spikework::Stack &s) {
    {
        Ptr<TimeSeries> ts = s.pop().as<TimeSeries>();
        Ptr<TimeSeries> kernel_ts = kernel->generate(1, kernel_length, kernel_dt);
        s.push(ts);
        s.push(kernel_ts);
    }
    {
        ConvWorker conv;
        conv.process(s);
    }
}


}

