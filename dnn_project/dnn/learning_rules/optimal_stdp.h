#pragma once

#include <dnn/protos/optimal_stdp.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/util/fastapprox/fastexp.h>
#include <dnn/neurons/srm_neuron.h>
#include <dnn/sim/global_ctx.h>
#include <dnn/sim/sim_info.h>
#include <dnn/util/fastapprox/fastpow.h>


namespace dnn {

template <typename C, typename S, typename N>
class LearningRule;

/*@GENERATE_PROTO@*/
struct OptimalStdpC : public Serializable<Protos::OptimalStdpC> {
    OptimalStdpC()
    : tau_c(100.0)
    , tau_mean(10000.0)
    , target_rate(10.0)
    , target_rate_factor(1.0)
    , weight_decay(0.0026)
    , learning_rate(0.01)
    , tau_mi_stat(30000.0)
    , tau_hebb(0.0)
    {}

    void serial_process() {
        begin()  <<
            "tau_c: " << tau_c << ", " <<
            "tau_mean: " << tau_mean << ", " <<
            "target_rate: " << target_rate << ", " <<
            "target_rate_factor: " << target_rate_factor << ", " <<
            "learning_rate: " << learning_rate << ", " <<
            "weight_decay: " << weight_decay  << ", " <<
            "tau_mi_stat: " << tau_mi_stat << ", " <<
            "tau_hebb: " << tau_hebb << Self::end;
        __target_rate = target_rate/1000.0;
    }

    double tau_c;
    double tau_mean;
    double target_rate;
    double __target_rate;
    double target_rate_factor;
    double weight_decay;
    double learning_rate;
    double tau_mi_stat;
    double tau_hebb;
};


/*@GENERATE_PROTO@*/
struct OptimalStdpState : public Serializable<Protos::OptimalStdpState>  {
    OptimalStdpState()
    : B(0.0), p_mean(0.0), mi_stat(0.0)
    {}

    void serial_process() {
        begin() << "p_mean: " << p_mean << ", "
                << "C: " << C << ", "
                << "B: " << B << ", "
                << "mi_stat: " << mi_stat << Self::end;
    }
    double p_mean;
    ActVector<double> C;
    double B;
    double mi_stat;
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

    void propagateSynapseSpike(const SynSpike &sp);

    inline double B_calc();
    void calculateDynamics(const Time& t);

};

}
