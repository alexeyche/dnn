
#include "entropy.h"


namespace dnn {


void EntropyKernel::usage(ostream &str) const {
    str << "spec: Entropy()\n";
    str << "desc: Cross entropy distance on time series\n";
}

double EntropyKernel::operator () (const vector<double> &x, const vector<double> &y) const {
    assert(x.size() == y.size());
    double acc = 0.0;
    // double x_norm = 0.0, y_norm = 0.0;
    for(size_t i=0; i<x.size(); ++i) {
        double denom = x[i]+y[i];
        if(denom>1e-08) {
            double pxy = x[i]/denom;
            double pyx = y[i]/denom;
            // cout << pxy << " " << pyx << "\n";
            if(pxy>1e-08) {
                acc += -pxy*log2(pxy);
            }
            if(pyx>1e-08) {
                acc += -pyx*log2(pyx);
            }
        }


        // x_norm += x[i]*x[i];
        // y_norm += y[i]*y[i];
    }
    // x_norm = sqrt(x_norm);
    // y_norm = sqrt(y_norm);
    // return acc/(x_norm*y_norm);
    return acc;
}



}