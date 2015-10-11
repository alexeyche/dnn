
#include "optimal_stdp.h"
#include "srm_methods.h"

namespace dnn {

void OptimalStdp::propagateSynapseSpike(const SynSpike &sp) {
    s.C.makeActive(sp.syn_id);
}


inline double OptimalStdp::B_calc() {
    // cout << s.p_mean << "\n";
    if( fabs(s.p_mean) < 0.00001 ) return(0);
    // stat.add("B0",  (double)n->fired() * log(n->getFiringProbability()/s.p_mean));
    // stat.add("B1", (n->getFiringProbability() - s.p_mean));
    // cout << (double)n->fired() << " * " << log(n->getFiringProbability()/s.p_mean) << " - (" << n->getFiringProbability() << " - " << s.p_mean << ")\n";
    return                        (( (double)n->fired() * log(n->getFiringProbability()/s.p_mean) - (n->getFiringProbability() - s.p_mean)) -  \
            c.target_rate_factor * ( (double)n->fired() * log(s.p_mean/c.__target_rate) - (s.p_mean - c.__target_rate)) );

}

void OptimalStdp::calculateDynamics(const Time& t) {
    s.p_mean += (-s.p_mean + (double)n->fired())/c.tau_mean;
    s.mi_stat += (
        -s.mi_stat +
        (
            SRMMethods::LLH_formula(n->fired(), n->getFiringProbability()) -
            SRMMethods::LLH_formula(n->fired(), s.p_mean)
        )
    )/c.tau_mi_stat;
    stat.add("mi_stat", s.mi_stat);

    //stat.add("p_mean", s.p_mean);
    if(GlobalCtx::inst().getSimInfo().pastTime < c.tau_mean) {
        return;
    }

    s.B = B_calc();
    stat.add("B", s.B);

    auto &syns = n->getMutSynapses();
    const auto &norm = n->getWeightNormalization().ifc();

    size_t i=0;
    for(auto &syni: syns) {
        auto &syn = syni.ref();
        const double &w = syn.weight();

        s.C.get(i) += SRMMethods::dLLH_dw(*n, syn, c.tau_hebb);  // not in propagateSpike because we need information about firing of neuron

        double decay_part = c.weight_decay * syn.fired() * syn.weight(); //* (s.p_mean*1000.0) * (s.p_mean*1000.0);
        double dw = norm.derivativeModulation(w) * c.learning_rate * (
            s.C.get(i) * s.B * norm.ltp(w) - decay_part * norm.ltd(w)
        );

        stat.add("C", i, s.C.get(i));
        // stat.add("decay_part", i, decay_part);
        // stat.add("dw", i, dw);
        stat.add("w", i, syn.weight());

        syn.mutWeight() += dw;

        s.C.get(i) += - s.C.get(i)/c.tau_c;
        ++i;
    }


    // auto C_id_it = s.C.ibegin();
    // while(C_id_it != s.C.iend()) {
    //     const size_t &syn_id = *C_id_it;
    //     auto &syn = syns.get(syn_id).ref();

    //     s.C[C_id_it] += SRMMethods::dLLH_dw(*n, syn);  // not in propagateSpike because we need information about firing of neuron

    //     double wp = fastpow(syn.weight(), 4.0);
    //     double cwp = fastpow(0.2, 4.0);
    //     double dw = (wp/(wp+cwp)) * c.learning_rate * ( s.C[C_id_it] * s.B - c.weight_decay * syn.fired() * syn.weight());


    //     syn.mutWeight() += dw;

    //     s.C[C_id_it] += - s.C[C_id_it]/c.tau_c;

    //     if(fabs(s.C[C_id_it]) < 0.0001) {
    //         s.C.setInactive(C_id_it);
    //     } else {
    //         ++C_id_it;
    //     }
    // }

    // if(stat.on()) {
    //     size_t i=0;
    //     for(auto &syn: syns) {
    //         stat.add("C", i, s.C.get(i));
    //         stat.add("w", i, syn.ref().weight());
    //         ++i;
    //     }
    //     stat.add("B", s.B);
    // }
}




}