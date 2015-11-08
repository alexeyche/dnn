#pragma once

#include "learning_rule.h"

#include <dnn/protos/voltage_stdp.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/fastapprox/fastexp.h>

#include <limits>

namespace dnn {


/*@GENERATE_PROTO@*/
struct VoltageStdpC : public Serializable<Protos::VoltageStdpC> {
    VoltageStdpC()
        : sigma_minus(0.0)
        , sigma_plus(1.2)        
        , tau_x(15.0)
        , tau_minus(10)
        , tau_plus(5)
        , a_plus(1.0)
        , a_minus(1.0)
        , learning_rate(0.04)
    {
    }

    void serial_process() {
        begin() << "sigma_minus: " << sigma_minus << ", " 
                << "sigma_plus: " << sigma_plus << ", " 
                << "tau_x: " << tau_x << ", "
                << "tau_plus: " << tau_plus << ", "
                << "tau_minus: " << tau_minus << ", "
                << "a_plus: " << a_plus << ", "
                << "a_minus: " << a_minus << ", "
                << "learning_rate: " << learning_rate << Self::end;
    }

    
    double sigma_minus;
    double sigma_plus;
    
    double tau_x;
    double tau_minus;
    double tau_plus;
    double a_plus;
    double a_minus;
    double learning_rate;
};


/*@GENERATE_PROTO@*/
struct VoltageStdpState : public Serializable<Protos::VoltageStdpState>  {
    VoltageStdpState()
        : u_minus(0.0), u_plus(0.0)
    {
    }

    void serial_process() {
        begin() << "u_minus: " << u_minus << ", "
                << "u_plus: " << u_plus << ", "
                << Self::end;
    }

    double u_minus;
    double u_plus;
    ActVector<double> x;
};

double rectifier(const double &&u) {
    return u < 0.0 ? 0.0 : u;
}

class VoltageStdp : public LearningRule<VoltageStdpC, VoltageStdpState, SpikeNeuronBase> {
public:
    const string name() const {
        return "VoltageStdp";
    }

    void reset() {
        s.u_minus = 0;
        s.u_plus = 0;
        s.x.resize(n->getSynapses().size());
        for(auto &v: s.x) {
            v = 0;
        }
    }

    void propagateSynapseSpike(const SynSpike &sp) {
        s.x[sp.syn_id] += 1.0/c.tau_x;
    }

    void calculateDynamics(const Time& t) {
        const double &u = n->getMembrane();

        s.u_minus += (-s.u_minus + u)/c.tau_minus;
        s.u_plus += (-s.u_plus + u)/c.tau_plus;

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
                    norm.ltp(w) * c.a_plus * s.x[x_id_it] * rectifier(u - c.sigma_plus) * rectifier(s.u_plus - c.sigma_minus) - 
                    norm.ltd(w) * c.a_minus * syn.fired() * rectifier(s.u_minus - c.sigma_minus)
                );
                if(syn.potential()<0) {
                    dw = -dw;
                }
                syn.mutWeight() += dw;

                s.x[x_id_it] += - s.x[x_id_it]/c.tau_x;
                ++x_id_it;
            }
        }
        

        if(stat.on()) {
            size_t i=0;

            for(auto &syn: syns) {
                stat.add("x", i, s.x.get(i));
                stat.add("w", i, syn.ref().weight());
                stat.add("ltd", i, norm.ltd(syn.ref().weight()));
                ++i;
            }
            stat.add("u_minus", s.u_minus);
            stat.add("u_plus", s.u_plus);
        }
    }

};

}

