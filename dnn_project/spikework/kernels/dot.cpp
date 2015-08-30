
#include "dot.h"


namespace dnn {


void DotKernel::usage(ostream &str) const {
    str << "spec: Dot()\n";
    str << "desc: Simple dot operation on column vectors from time series. Result normalized by norm of vectors\n";
}

double DotKernel::operator () (const vector<double> &x, const vector<double> &y) const {
    assert(x.size() == y.size());
    double acc = 0.0;
    double x_norm = 0.0, y_norm = 0.0;
    for(size_t i=0; i<x.size(); ++i) {
        acc += x[i]*y[i];
        x_norm += x[i]*x[i];
        y_norm += y[i]*y[i];
    }
    x_norm = sqrt(x_norm);
    y_norm = sqrt(y_norm);
    return acc/(x_norm*y_norm);
}



}