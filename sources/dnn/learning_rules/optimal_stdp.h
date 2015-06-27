#pragma once


#include "learning_rule.h"
#include "srm_methods.h"

#include <dnn/protos/generated.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/fastapprox/fastexp.h>
#include <dnn/neurons/srm_neuron.h>
#include <dnn/sim/global_ctx.h>
#include <dnn/sim/sim_info.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct OptimalStdpC : public Serializable<Protos::OptimalStdpC> {
    OptimalStdpC() 
    : tau_c(100.0)
    , tau_mean(10000.0)
    , target_rate(10.0)
    , target_rate_factor(1.0)
    , weight_decay(0.0026)
    , learning_rate(0.01)
    {}

    void serial_process() {
        begin()  <<
            "tau_c: " << tau_c << ", " <<
            "tau_mean: " << tau_mean << ", " <<
            "target_rate: " << target_rate << ", " <<
            "target_rate_factor: " << target_rate_factor << ", " <<
            "learning_rate: " << learning_rate << ", " <<
            "weight_decay: " << weight_decay  <<  Self::end;
        __target_rate = target_rate/1000.0;
    }

    double tau_c;
    double tau_mean;
    double target_rate;
    double __target_rate;
    double target_rate_factor;
    double weight_decay;
    double learning_rate;
};


/*@GENERATE_PROTO@*/
struct OptimalStdpState : public Serializable<Protos::OptimalStdpState>  {
    OptimalStdpState() 
    : B(0.0), p_mean(0.0)
    {}

    void serial_process() {
        begin() << "p_mean: " << p_mean << ", " 
                << "C: " << C << ", " 
                << "B: " << B << Self::end;
    }
    double p_mean;
    ActVector<double> C;
    double B;

};


class OptimalStdp : public LearningRule<OptimalStdpC, OptimalStdpState, SRMNeuron> {
public:
    const string name() const {
        return "OptimalStdp";
    }

    void reset() {
        s.B = 0.0;
        s.C.resize(n->getSynapses().size());
        fill(s.C.begin(), s.C.end(), 0.0);
        s.p_mean = 0.0;        
    }

    void propagateSynapseSpike(const SynSpike &sp) {
        s.C[sp.syn_id] += SRMMethods::dLLH_dw(*n, n->getSynapses().get(sp.syn_id).ref());
    }
    
    inline double B_calc() const {
        if( fabs(s.p_mean) < 0.00001 ) return(0);
        return                        (( n->fired() * log(n->getFiringProbability()/s.p_mean) - (n->getFiringProbability() - s.p_mean)) -  \
                c.target_rate_factor * ( n->fired() * log(s.p_mean/c.__target_rate) - (s.p_mean - c.__target_rate)) );

    }
    
    void calculateDynamics(const Time& t) {
        s.p_mean += (-s.p_mean + 1.0 ? n->fired() : 0.0)/c.tau_mean;
        stat.add("p_mean", s.p_mean);
        if(GlobalCtx::inst().getSimInfo().pastTime < c.tau_mean) { 
            return;
        }

        auto &syns = n->getMutSynapses();
        
        auto C_id_it = s.C.ibegin();
        while(C_id_it != s.C.iend()) {            
            if(fabs(s.C[C_id_it]) < 0.0001) {
                s.C.setInactive(C_id_it);
            } else {
                const size_t &syn_id = *C_id_it;
                auto &syn = syns.get(syn_id).ref();
                
                s.C[C_id_it] += - s.C[C_id_it]/c.tau_c;
                ++C_id_it;
            }
        }

        if(stat.on()) {
            size_t i=0; 
            for(auto &syn: syns) {
                stat.add("C", i, s.C.get(i));
                stat.add("w", i, syn.ref().weight());
                ++i;
            }
            stat.add("B", s.B);        
        }
    }
    
    

};

}
