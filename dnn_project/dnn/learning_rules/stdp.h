#pragma once


#include "learning_rule.h"
#include <dnn/protos/stdp.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/fastapprox/fastexp.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct StdpC : public Serializable<Protos::StdpC> {
    StdpC()
    : tau_plus(30.0)
    , tau_minus(50.0)
    , a_plus(1.0)
    , a_minus(1.5)
    , learning_rate(1.0)
    {}

    void serial_process() {
        begin() << "tau_plus: " << tau_plus << ", "
                << "tau_minus: " << tau_minus << ", "
                << "a_plus: " << a_plus << ", "
                << "a_minus: " << a_minus << ", "
                << "learning_rate: " << learning_rate << Self::end;
    }

    double tau_plus;
    double tau_minus;
    double a_plus;
    double a_minus;
    double learning_rate;
};


/*@GENERATE_PROTO@*/
struct StdpState : public Serializable<Protos::StdpState>  {
    StdpState()
    : y(0.0)
    {}

    void serial_process() {
        begin() << "y: "        << y << ", "
                << "x: "        << x << Self::end;
    }

    double y;
    ActVector<double> x;
};


class Stdp : public LearningRule<StdpC, StdpState, SpikeNeuronBase> {
public:

    void reset() {
        s.y = 0;
        s.x.resize(n->getSynapses().size());
        for(auto &v: s.x) {
            v = 0;
        }
    }

    void propagateSynapseSpike(const SynSpike &sp) {
        s.x[sp.syn_id] += 1;
    }

    void calculateDynamics(const Time& t) {
        if(n.ref().fired()) {
            s.y += 1;
        }
        auto &syns = n->getMutSynapses();
        const auto &norm = n->getWeightNormalization().ifc();

        auto x_id_it = s.x.ibegin();
        while(x_id_it != s.x.iend()) {
            if(fabs(s.x[x_id_it]) < 0.0001) {
                s.x.setInactive(x_id_it);
            } else {
                const size_t &syn_id = *x_id_it;
                auto &syn = syns.get(syn_id).ref();
                const double &w = syn.weight();

                double dw = c.learning_rate * norm.derivativeModulation(w) * (
                    c.a_plus  * s.x[x_id_it] * n->fired() * norm.ltp(w) - \
                    c.a_minus * s.y * syn.fired() * norm.ltd(w)
                );
                if(syn.potential()<0) {
                    dw = -dw;
                }
                syn.mutWeight() += dw;


                s.x[x_id_it] += - s.x[x_id_it]/c.tau_plus;
                ++x_id_it;
            }
        }
        s.y += - s.y/c.tau_minus;

        if(stat.on()) {
            size_t i=0;
            for(auto &syn: syns) {
                stat.add("x", i, s.x.get(i));
                stat.add("w", i, syn.ref().weight());
                ++i;
            }
            stat.add("y", s.y);
        }
    }

};

}
