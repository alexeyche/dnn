#pragma once

#include "synapse.h"
#include <dnn/protos/static_synapse.pb.h>
#include <dnn/io/serialize.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct StaticSynapseC : public Serializable<Protos::StaticSynapseC>  {
    StaticSynapseC() : psp_decay(15.0), amp(1.0) {}
    
    void serial_process() {
        begin() << "psp_decay: " << psp_decay << ", " \
                << "amp: "       << amp       << Self::end;
    }
    
    double psp_decay;
    double amp;
};

/*@GENERATE_PROTO@*/
struct StaticSynapseState : public Serializable<Protos::StaticSynapseState>  {
    StaticSynapseState() {}

    void serial_process() {
        begin() << Self::end;
    }
};

class StaticSynapse : public Synapse<StaticSynapseC, StaticSynapseState> {
public:
    const string name() const {
        return "StaticSynapse";
    }
    void reset() {
        mutPotential() = 0;
    }
    void propagateSpike() {
        mutPotential() += c.amp;
    }
    void calculateDynamics(const Time &t) {
        stat.add("x", potential());
        mutPotential() += - t.dt * potential()/c.psp_decay;
    }
};




}
