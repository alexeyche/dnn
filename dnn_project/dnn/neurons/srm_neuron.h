#pragma once

#include <dnn/util/fastapprox/fastexp.h>
#include "spike_neuron.h"
#include <dnn/protos/srm_neuron.pb.h>
#include <dnn/io/serialize.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct SRMNeuronC : public Serializable<Protos::SRMNeuronC> {
    SRMNeuronC()
    : u_rest(0.0)
    , amp_adapt(1.0)
    , amp_refr(-100.0)
    , tau_refr(2.0)
    , tau_adapt(50.0)
    {}

    void serial_process() {
        begin()  << "u_rest: " << u_rest << ", " <<
    			    "amp_refr: " << amp_refr << ", " <<
                    "amp_adapt: " << amp_adapt << ", " <<
    				"tau_refr: " << tau_refr << ", " <<
        			"tau_adapt: " << tau_adapt  << Self::end;
    }

    double u_rest;
    double amp_refr;
    double amp_adapt;
    double tau_refr;
    double tau_adapt;
};


/*@GENERATE_PROTO@*/
struct SRMNeuronState : public Serializable<Protos::SRMNeuronState>  {
    SRMNeuronState()
    : M(1.0)
    , gr(0.0)
    , ga(0.0)
    {}

    void serial_process() {
        begin() << "gr: " << gr << ", "
                << "ga: " << ga << ", "
                << "M: " << M << Self::end;
    }

    double M;
    double gr;
    double ga;
};


class SRMNeuron : public SpikeNeuron<SRMNeuronC, SRMNeuronState> {
public:
    const string name() const {
        return "SRMNeuron";
    }

    void reset() {
        s = SRMNeuronState();
    }

    void calculateDynamics(const Time& t, const double &Iinput, const double &Isyn) {
        membrane() = c.u_rest + Iinput + Isyn;
        s.M = fastexp(-(s.gr+s.ga));
        firingProbability() = act_f.ifc().prob(membrane()) * s.M;

        // membrane()*=s.M;

        if(firingProbability() > getUnif()) {
            setFired(true);
            s.gr += c.amp_refr;
            s.ga += c.amp_adapt;
        }
        s.gr += - s.gr/c.tau_refr;
        s.ga += - s.ga/c.tau_adapt;

        stat.add("u", membrane());
        stat.add("p", firingProbability());
        stat.add("M", s.M);
        stat.add("ga", s.ga);
    }

    const double& getProbabilityModulation() const {
        return s.M;
    }
};

}
