
#include "anova_dot.h"

#include <dnn/util/func_param_parser.h>
#include <dnn/util/fastapprox/fastpow.h>
#include <dnn/util/fastapprox/fastexp.h>

namespace dnn {


void AnovaDotKernel::usage(ostream &str) const {
    str << "spec: AnovaDot(sigma = " << o.sigma << ", power = " << o.power << ")\n";
    str << "desc: Anova radial basis function, performs well in multidimensional regression problems.\n";
}

double AnovaDotKernel::operator () (const vector<double> &x, const vector<double> &y) const {
    assert(x.size() == y.size());
    double acc = 0.0;
    for(size_t i=0; i<x.size(); ++i) {
        acc += fastexp(-o.sigma*(x[i] - y[i])*(x[i] - y[i]));
    }
    return fastpow(acc, o.power);
}

void AnovaDotKernel::processSpec(const string &spec) {
    L_DEBUG << "AnovaDotKernel, processing spec: " << spec;
    FuncParamParser::TParserMap m;
    m["sigma"] = FuncParamParser::genDoubleParser(o.sigma);
    m["power"] = FuncParamParser::genDoubleParser(o.power);
    FuncParamParser::parse(spec, m);
}

}