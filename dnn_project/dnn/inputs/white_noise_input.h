#pragma once

#include "input.h"

#include <dnn/protos/white_noise_input.pb.h>
#include <dnn/io/stream.h>
#include <dnn/util/log/log.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct WhiteNoiseInputC : public Serializable<Protos::WhiteNoiseInputC> {
    WhiteNoiseInputC() : mean(0.0), sd(1.0), T(100.0) {}

    double mean;
    double sd;

    double T;

    void serial_process() {
        begin() << "mean: " << mean << ", " << "sd: " << sd << ", " << "T: " << T << Self::end;
    }
};

/*@GENERATE_PROTO@*/
struct WhiteNoiseInputState : public Serializable<Protos::WhiteNoiseInputState> {
    WhiteNoiseInputState() : rng_state(0.0), cache(false) {}

    double rng_state;
    bool cache;

    void serial_process() {
        begin() << "rng_state: " << rng_state << ", " << "cache: " << cache << Self::end;
    }
};


class WhiteNoiseInput : public Input<WhiteNoiseInputC, WhiteNoiseInputState> {
public:
    typedef Input<WhiteNoiseInputC, WhiteNoiseInputState> Parent;

    const string name() const {
        return "WhiteNoiseInput";
    }
    void reset() {
        s = WhiteNoiseInputState();
    }

    void init() {
        GlobalCtx::inst().setSimDuration(c.T);
    }

    const double genUnitNoise() {
        if(!s.cache) {
            double U = getUnif();
            double V = getUnif();
            s.rng_state = sqrt(-2*log(U)) * cos(2*PI*V);
            s.cache = true;
            return sqrt(-2*log(U)) * sin(2*PI*V);
        } else {
            s.cache = false;
            return s.rng_state;
        }
    }

    const double getValue(const Time &t) {
        return c.mean + genUnitNoise()*c.sd;
    }


};



}
