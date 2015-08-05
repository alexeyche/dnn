#pragma once

#include "synapse.h"
#include <dnn/protos/std_synapse.pb.h>
#include <dnn/io/serialize.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct STDSynapseC : public Serializable<Protos::STDSynapseC>  {
    STDSynapseC() : psp_decay(2.0), amp(1.0), gamma(0.65), tau_d(400.0) {}
    
    void serial_process() {
        begin() << "psp_decay: " << psp_decay << ", "
                << "amp: "       << amp     << ", " 
                << "gamma: "     << gamma   << ", " 
                << "tau_d: "     << tau_d   << Self::end;
    }
    
    double psp_decay;
    double amp;
    double gamma;
    double tau_d;
};

/*@GENERATE_PROTO@*/
struct STDSynapseState : public Serializable<Protos::STDSynapseState>  {
    STDSynapseState() : res(1.0) {}

    void serial_process() {
        begin() << "res: "  << res << Self::end;
    }

    double res;
};

class STDSynapse : public Synapse<STDSynapseC, STDSynapseState> {
public:
    const string name() const {
        return "STDSynapse";
    }
    
    void reset() {
        mutPotential() = 0.0;
        s.res = 1.0;
    }

    void propagateSpike() {
        mutPotential() += c.amp * s.res;
        s.res -= (1 - c.gamma) * s.res;
    }
    void calculateDynamics(const Time &t) {
        stat.add("x", potential());
        stat.add("res", s.res);
        
        mutPotential() += - t.dt * potential()/c.psp_decay;   
        s.res += (1 - s.res)/c.tau_d;
    }
};




}
