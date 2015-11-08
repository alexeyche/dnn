#pragma once


#include "spike_neuron.h"
#include <dnn/protos/leaky_integrate_and_fire.pb.h>
#include <dnn/io/serialize.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct LeakyIntegrateAndFireC : public Serializable<Protos::LeakyIntegrateAndFireC> {
    LeakyIntegrateAndFireC()
    :
      tau_m(5.0)
    , rest_pot(0.0)
    , tau_ref(2.0)
    , noise(0.0)
    {}

    void serial_process() {
        begin() << "tau_m: " << tau_m << ", "
                << "rest_pot: " << rest_pot << ", "
                << "tau_ref: " << tau_ref << ", "
                << "noise: " << noise << Self::end;
    }

    double tau_m;
    double rest_pot;
    double tau_ref;
    double noise;
};


/*@GENERATE_PROTO@*/
struct LeakyIntegrateAndFireState : public Serializable<Protos::LeakyIntegrateAndFireState>  {
    LeakyIntegrateAndFireState()
        : ref_time(0.0)
    {}

    void serial_process() {
        begin() << "ref_time: " << ref_time << Self::end;
    }
    double ref_time;
};


class LeakyIntegrateAndFire : public SpikeNeuron<LeakyIntegrateAndFireC, LeakyIntegrateAndFireState> {
public:
    const string name() const {
        return "LeakyIntegrateAndFire";
    }

    void reset() {
        membrane() = c.rest_pot;
        firingProbability() = 0.0;
        s.ref_time = 0.0;
    }

    void calculateDynamics(const Time& t, const double &Iinput, const double &Isyn) {
        if(s.ref_time < 0.001) {
            membrane() += t.dt * ( - membrane()  + c.noise*getNorm() + Iinput + Isyn) / c.tau_m;
            firingProbability() = act_f.ifc().prob(membrane());

            if(firingProbability() > getUnif()) {
                setFired(true);
                membrane() = c.rest_pot;
                s.ref_time = c.tau_ref;
            }
        } else {
            s.ref_time -= t.dt;
        }
        stat.add("u", membrane());
        stat.add("p", firingProbability());
    }
};

}
