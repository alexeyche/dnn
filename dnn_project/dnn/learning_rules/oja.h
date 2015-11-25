#pragma once


#include "learning_rule.h"
#include <dnn/protos/oja.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/fastapprox/fastexp.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct OjaC : public Serializable<Protos::OjaC> {
    OjaC()
    : tau_x(100.0)
    , tau_y(100.0)
    , learning_rate(1.0)
    {}

    void serial_process() {
        begin() << "tau_x: " << tau_x << ", "
                << "tau_y: " << tau_y << ", "
                << "learning_rate: " << learning_rate << Self::end;
    }

    double tau_x;
    double tau_y;
    double learning_rate;
};


/*@GENERATE_PROTO@*/
struct OjaState : public Serializable<Protos::OjaState>  {
    OjaState()
    : y(0.0)
    {}

    void serial_process() {
        begin() << "y: "        << y << ", "
                << "x: "        << x << Self::end;
    }

    double y;
    ActVector<double> x;
};


class Oja : public LearningRule<OjaC, OjaState, SpikeNeuronBase> {
public:
    const string name() const {
        return "Oja";
    }

    void reset() {
        s.y = 0;
        auto syns = n->getSynapses();
        s.x.resize(syns.size());        
    }

    void propagateSynapseSpike(const SynSpike &sp) {
        s.x[sp.syn_id] += 1.0/c.tau_x;
    }

    void calculateDynamics(const Time& t) {
        if(n.ref().fired()) {
            s.y += (double)n->fired()/c.tau_y;
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
                    norm.ltp(w) * s.y * s.x[x_id_it] - norm.ltd(w) * s.y * s.y * w
                );
                if(syn.isInhibitory()) {
                    dw = -dw;
                }
                syn.mutWeight() += dw;

                s.x[x_id_it] += - s.x[x_id_it]/c.tau_x;
                ++x_id_it;
            }
        }
        s.y += - s.y/c.tau_y;

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
