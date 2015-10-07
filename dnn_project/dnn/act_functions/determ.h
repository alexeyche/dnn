#pragma once


#include "act_function.h"
#include <dnn/protos/determ.pb.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct DetermC : public Serializable<Protos::DetermC> {
    DetermC() : threshold(1.0) {}

    void serial_process() {
        begin() << "threshold: " << threshold << Self::end;
    }


    double threshold;
};


class Determ : public ActFunction<DetermC> {
public:
    const string name() const {
        return "Determ";
    }
    double prob(const double &u) {
        if(u >= c.threshold) {
            return 1.0;
        }
        return 0.0;
    }
    double probDeriv(const double &u) {
        return 0.0;
    }
};



}
