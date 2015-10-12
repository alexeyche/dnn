#pragma once


#include "weight_normalization.h"

#include <dnn/protos/sliding_ltd.pb.h>
#include <dnn/util/fastapprox/fastpow.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct SlidingLtdC : public Serializable<Protos::SlidingLtdC> {
    SlidingLtdC()
    : power(3.0)
    , modulation(1.0)
    , target_rate(5.0)
    , tau_mean(100)
    , min_weight(0.0)
    , max_weight(1.0)
    {
    }

    void serial_process() {
        begin() << "power: " << power << ", "
                << "target_rate: " << target_rate << ", "
                << "modulation: " << modulation << ", "
                << "tau_mean: " << tau_mean << ", "
                << "min_weight: " << min_weight << ", "
                << "max_weight: " << max_weight << Self::end;

    	__target_rate = 1.0/fastpow(target_rate, power);
    }

    double power;
    double modulation;
    double target_rate;
    double __target_rate;
    double tau_mean;
    double min_weight;
    double max_weight;
};


/*@GENERATE_PROTO@*/
struct SlidingLtdState : public Serializable<Protos::SlidingLtdState>  {
    SlidingLtdState()
    : p_mean(0.0)
    {}

    void serial_process() {
        begin() << "p_mean: " << p_mean << Self::end;
    }

    double p_mean;
};


class SlidingLtd : public WeightNormalization<SlidingLtdC, SlidingLtdState> {
public:
    const string name() const {
        return "SlidingLtd";
    }
    double ltd(const double &w) {
        if(GlobalCtx::inst().getSimInfo().pastTime < 5*c.tau_mean) {
            return c.modulation;
        }
        return c.modulation * fastpow(1000.0*s.p_mean, c.power) * c.__target_rate;
    }

    void calculateDynamics(const Time &t) {
        s.p_mean += (-s.p_mean + (double)n->fired())/c.tau_mean;
    }

    double derivativeModulation(const double &w) {
        if((w >= c.max_weight)||(w <= c.min_weight)) {
            return 0.0;
        }
        return 1.0;
    }
};



}
