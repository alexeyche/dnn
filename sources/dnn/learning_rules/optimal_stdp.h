#pragma once


#include "learning_rule.h"
#include "srm_methods.h"

#include <dnn/protos/generated.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/fastapprox/fastexp.h>
#include <dnn/neurons/srm_neuron.h>
#include <dnn/sim/global_ctx.h>
#include <dnn/sim/sim_info.h>
#include <dnn/util/fastapprox/fastpow.h>

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
    }

    void propagateSynapseSpike(const SynSpike &sp) {
        s.C.makeActive(sp.syn_id);
    }
    
    inline double B_calc() const {
        // cout << s.p_mean << "\n";
        if( fabs(s.p_mean) < 0.00001 ) return(0);
        // cout << (double)n->fired() << " * " << log(n->getFiringProbability()/s.p_mean) << " - (" << n->getFiringProbability() << " - " << s.p_mean << ")\n"; 
        return                        (( (double)n->fired() * log(n->getFiringProbability()/s.p_mean) - (n->getFiringProbability() - s.p_mean)) -  \
                c.target_rate_factor * ( (double)n->fired() * log(s.p_mean/c.__target_rate) - (s.p_mean - c.__target_rate)) );

    }
    
    void calculateDynamics(const Time& t) {
        s.p_mean += (-s.p_mean + (double)n->fired())/c.tau_mean;
        stat.add("p_mean", s.p_mean);
        if(GlobalCtx::inst().getSimInfo().pastTime < 10*c.tau_mean) { 
            return;
        }

        s.B = B_calc();
        auto &syns = n->getMutSynapses();
        
        auto C_id_it = s.C.ibegin();
        while(C_id_it != s.C.iend()) {
            const size_t &syn_id = *C_id_it;
            auto &syn = syns.get(syn_id).ref();

            s.C[C_id_it] += SRMMethods::dLLH_dw(*n, syn);  // not in propagateSpike because we need information about firing of neuron
            
            double wp = fastpow(syn.weight(), 4.0);
            double cwp = fastpow(0.2, 4.0);            
            double dw = (wp/(wp+cwp)) * c.learning_rate * ( s.C[C_id_it] * s.B - c.weight_decay * syn.fired() * syn.weight());
            

            syn.mutWeight() += dw;

            s.C[C_id_it] += - s.C[C_id_it]/c.tau_c;

            if(fabs(s.C[C_id_it]) < 0.0001) {
                s.C.setInactive(C_id_it);
            } else {
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
