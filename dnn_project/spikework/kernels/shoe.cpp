
#include "shoe.h"
#include <dnn/util/func_param_parser.h>
#include <spikework/kernel_factory.h>
#include <dnn/util/fastapprox/fastexp.h>


namespace dnn {


void ShoeKernel::usage(ostream &str) const {
    str << "spec: Shoe(kernel_spec, sigma = " << o.sigma << ")\n";
    str << "desc: Shoenberg kernel, based on another kernel specification which must be pointed as first parameter\n";
}

double ShoeKernel::operator () (const vector<double> &x, const vector<double> &y) const {
    const IKernel &k = kernel.ref();
    return fastexp( - o.sigma*(k(x, x) - 2 * k(x, y) + k(y, y)));
}

void ShoeKernel::processSpec(const string &spec) {
    L_DEBUG << "ShoeKernel, processing spec: " << spec;
    FuncParamParser::TParserMap m;
    m["kernel_spec"] = FuncParamParser::genStringParser(o.kernel_spec);
    m["sigma"] = FuncParamParser::genDoubleParser(o.sigma);
    FuncParamParser::parse(spec, m);
    if(o.kernel_spec.empty()) {
        throw dnnException() << "Must define kernel specification as first parameter\n";
    }
    kernel = KernelFactory::inst().createKernel(o.kernel_spec);
}

}