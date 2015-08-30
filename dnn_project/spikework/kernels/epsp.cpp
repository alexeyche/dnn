
#include "epsp.h"

#include <dnn/util/log/log.h>

namespace dnn {


Ptr<TimeSeries> EpspKernel::operator () (Ptr<TimeSeries> x) const {
    return convolve(x, o.length, o.dt);
}

void EpspKernel::usage(ostream &str) const {
    str << "spec: Epsp(tau_decay = 10.0, tau_rise = 0.0, length = 100, dt = 1.0)\n";
    str << "desc: Kernel that makes convolution with exponential rise and decay, or decay only if rise is zero:\n";
    str << "\tEpsp(5, 15), Epsp(15), accordingly\n";

}

void EpspKernel::processSpec(const string &spec) {
    L_DEBUG << "EpspKernel, processing spec: " << spec;
    FuncParamParser::TParserMap m;
    m["tau_decay"] = FuncParamParser::genDoubleParser(o.tau_decay);
    m["tau_rise"] = FuncParamParser::genDoubleParser(o.tau_rise);
    m["length"] = FuncParamParser::genDoubleParser(o.length);
    m["dt"] = FuncParamParser::genDoubleParser(o.dt);
    FuncParamParser::parse(spec, m);

    if(o.tau_rise>0.0001) {
        if(fabs(o.tau_rise - o.tau_decay) < 1e-06) {
            o.tau_decay += 1e-05;
        }
        fun = [=](double t) {
            return (1/(1-o.tau_rise/o.tau_decay)) * (exp(-t/o.tau_decay) - exp(-t/o.tau_rise));
        };
    } else {
        fun = [=](double t) {
            return exp(-t/o.tau_decay);
        };
    }
}



}