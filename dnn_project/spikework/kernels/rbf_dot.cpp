
#include "rbf_dot.h"

#include <dnn/util/func_param_parser.h>
#include <dnn/util/fastapprox/fastexp.h>


namespace dnn {


void RbfDotKernel::usage(ostream &str) const {
    str << "spec: RbfDot(sigma = " << o.sigma << ")\n";
    str << "desc: Gaussian radial basis function, which is a general purpose kernel\n";
}

double RbfDotKernel::operator () (const vector<double> &x, const vector<double> &y) const {
    assert(x.size() == y.size());
    double acc = 0.0;
    // double x_norm = 0.0, y_norm = 0.0;
    for(size_t i=0; i<x.size(); ++i) {
        acc += o.sigma*(x[i]-y[i])*(x[i] - y[i]);
        // x_norm += x[i]*x[i];
        // y_norm += y[i]*y[i];
    }
    acc = fastexp(-acc);
    // x_norm = sqrt(x_norm);
    // y_norm = sqrt(y_norm);
    // return acc/(x_norm*y_norm);
    return acc;
}

void RbfDotKernel::processSpec(const string &spec) {
    L_DEBUG << "RbfDotKernel, processing spec: " << spec;
    FuncParamParser::TParserMap m;
    m["sigma"] = FuncParamParser::genDoubleParser(o.sigma);
    FuncParamParser::parse(spec, m);
}

}