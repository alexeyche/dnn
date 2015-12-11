#pragma once

#include "synapse.h"
#include <dnn/protos/hedonistic_synapse.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/sim/global_ctx.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct HedonisticSynapseC : public Serializable<Protos::HedonisticSynapseC>  {
    HedonisticSynapseC()
        : psp_decay(10.0)
        , amp(1.0)
        , tau_ref(350)
        , delta_catalyst(0.25)
        , tau_catalyst(500)
        , tau_eligibility(750)
        , learning_rate(0.25) {}

    void serial_process() {
        begin() << "psp_decay: "        << psp_decay       << ", " \
                << "amp: "              << amp             << ", " \
                << "tau_ref: "          << tau_ref         << ", " \
                << "delta_catalyst: "   << delta_catalyst  << ", " \
                << "tau_catalyst: "     << tau_catalyst    << ", " \
                << "tau_eligibility: "  << tau_eligibility << ", " \
                << "learning_rate: "    << learning_rate   << Self::end;

        double zero = numeric_limits<double>::epsilon();
        if(fabs(tau_ref) <= zero) {
            throw dnnException() << "Time constant tau_ref is too small, division by zero\n";
        }
        if(fabs(tau_catalyst) <= zero) {
            throw dnnException() << "Time constant tau_catalyst is too small, division by zero\n";
        }
        if(fabs(tau_eligibility) <= zero) {
            throw dnnException() << "Time constant tau_eligibility is too small, division by zero\n";
        }
    }

    double psp_decay;
    double amp;
    double tau_ref;
    double delta_catalyst;
    double tau_catalyst;
    double tau_eligibility;
    double learning_rate;
};

/*@GENERATE_PROTO@*/
struct HedonisticSynapseState : public Serializable<Protos::HedonisticSynapseState>  {
    HedonisticSynapseState()
        : refractory(0)
        , probability(0.5)
        , catalyst(0.0)
        , prob_weight(0)
        , eligibility_trace(0) {}

    void serial_process() {
        begin() << "refractory: "   << refractory   << ", " \
                << "probability: "  << probability  << ", " \
                << "catalyst: "     << catalyst     << ", " \
                << "prob_weight: "  << prob_weight  << ", " \
                << "eligibility_trace: "  << eligibility_trace  << Self::end;
    }

    double refractory;
    double probability;
    double catalyst;            // c
    double prob_weight;         // q
    double eligibility_trace;   // e
};

class HedonisticSynapse : public Synapse<HedonisticSynapseC, HedonisticSynapseState> {
public:
    void reset() {
        mutAmplitude() = c.amp;
        mutPotential() = 0;
        s.refractory = 0;
    }

    void propagateSpike() {
        if (s.refractory < 0.1) {
            s.probability = 1.0 / (1.0 + exp(-s.prob_weight - s.catalyst));
            if (s.probability > getUnif()) {
                // release
                mutPotential() += c.amp;
                s.eligibility_trace += 1 - s.probability;
                s.refractory = c.tau_ref;
                s.catalyst += c.delta_catalyst;
            }
            else {
                // failure
                s.eligibility_trace += -s.probability;
            }
        }
    }

    void calculateDynamics(const Time &t) {
        stat.add("x", potential());
        stat.add("eligibility_trace", s.eligibility_trace);
        stat.add("prob_weight", s.prob_weight);

        mutPotential() += - t.dt * potential()/c.psp_decay;

        s.eligibility_trace += - t.dt * s.eligibility_trace/c.tau_eligibility;
        s.catalyst += - t.dt * s.catalyst/c.tau_catalyst;
        s.refractory += - t.dt * s.refractory/c.tau_ref;
        s.prob_weight += c.learning_rate * s.eligibility_trace * GlobalCtx::inst().getReward();
    }
};

}
