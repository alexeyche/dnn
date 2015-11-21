#pragma once


#include "weight_normalization.h"

#include <dnn/protos/strict_bound.pb.h>
#include <dnn/util/fastapprox/fastpow.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct StrictBoundC : public Serializable<Protos::StrictBoundC> {
    StrictBoundC()
    : unit(1.0)
    , power(2.0)
    {
    }

    void serial_process() {
        begin() << "unit: " << unit << ", "
                << "power: " << power << Self::end;
    }

    double unit;
    double power;
};


class StrictBound : public WeightNormalization<StrictBoundC> {
public:
    const string name() const {
        return "StrictBound";
    }
    double derivativeModulation(const double &w) {
        if((w >= c.unit)||(w <= 0.0)) {
            return 0.0;
        }
        return 1.0;
    }
    void calculateDynamics(const Time &t) {
        auto &syns = n->getMutSynapses();
        double denominator = 0.0;
        for(auto s: syns) {
            denominator += fastpow(s.ref().weight(), c.power);
        }
        double mod = c.unit/fastpow(denominator, 1.0/c.power);
        for(auto s: syns) {
            s.ref().mutWeight() = s.ref().weight()*mod;
        }
    }
};



}
