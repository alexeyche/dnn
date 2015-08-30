#pragma once

#include <spikework/kernel.h>


namespace dnn {

class DotKernel : public Kernel<> {
public:
    void usage(ostream &str) const;

    double operator () (const vector<double> &x, const vector<double> &y) const;
};


}