#pragma once

#include "synapse.h"
#include <dnn/protos/hedonistic_synapse.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/sim/global_ctx.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct HedonisticSynapseC : public Serializable<Protos::HedonisticSynapseC>  {
    HedonisticSynapseC() : psp_decay(10.0), amp(1.0), delta_catalyst(0.25), tau_catalyst(500), tau_trace(750), learning_rate(0.25) {}

    void serial_process() {
        begin() << "psp_decay: "        << psp_decay       << ", " \
                << "amp: "              << amp             << ", " \
                << "delta_catalyst: "   << delta_catalyst  << ", " \
                << "tau_catalyst: "     << tau_catalyst    << ", " \
                << "tau_trace: "        << tau_trace       << ", " \
                << "learning_rate: "    << learning_rate   << Self::end;
    }

    double psp_decay;
    double amp;
    double delta_catalyst;
    double tau_catalyst;
    double tau_trace;
    double learning_rate;
};

/*@GENERATE_PROTO@*/
struct HedonisticSynapseState : public Serializable<Protos::HedonisticSynapseState>  {
    HedonisticSynapseState() : probability(0.5), catalyst(0.0), prob_weight(0), spike_trace(0) {}

    void serial_process() {
        begin() << "probability: "  << probability  << ", " \
                << "catalyst: "     << catalyst     << ", " \
                << "prob_weight: "  << prob_weight  << ", " \
                << "spike_trace: "  << spike_trace  << Self::end;
    }
    
    double probability;
    double catalyst;    // c
    double prob_weight; // q
    double spike_trace; // e
};

class HedonisticSynapse : public Synapse<HedonisticSynapseC, HedonisticSynapseState> {
public:
    const string name() const {
        return "HedonisticSynapse";

    }
    void reset() {
        mutAmplitude() = c.amp;
        mutPotential() = 0;
    }

    void propagateSpike() {

        s.probability = 1.0 / (1.0 + exp(-s.prob_weight - s.catalyst));
        if (s.probability > getUnif()) {
            // success
            mutPotential() += c.amp;
            s.spike_trace += 1 - s.probability;
        }
        else {
            s.spike_trace += -s.probability;
        }
        s.catalyst += c.delta_catalyst;
    }

    void calculateDynamics(const Time &t) {
        stat.add("x", potential());
        stat.add("spike_trace", s.spike_trace);
        stat.add("prob_weight", s.prob_weight);

        mutPotential() += - t.dt * potential()/c.psp_decay;

        s.spike_trace += -t.dt * s.spike_trace/c.tau_trace;
        s.catalyst += -t.dt * s.catalyst/c.tau_catalyst;
        s.prob_weight += c.learning_rate * s.spike_trace * GlobalCtx::inst().getReward();

    }
};




}
