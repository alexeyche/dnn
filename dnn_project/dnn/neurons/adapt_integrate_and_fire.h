#pragma once


#include "spike_neuron.h"
#include <dnn/protos/adapt_integrate_and_fire.pb.h>
#include <dnn/io/serialize.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct AdaptIntegrateAndFireC : public Serializable<Protos::AdaptIntegrateAndFireC> {
    AdaptIntegrateAndFireC()
    :
      tau_m(1.0)
    , rest_pot(-70.0)
    , tau_ref(2.0)
    , noise(1.5)
    , tau_adapt(80.0)
    , kd(30.0)
    , vK(-80.0)
    , adapt_amp(0.2)
    , gKCa(5.0)
    {}

    void serial_process() {
        begin() << "tau_m: " << tau_m << ", "
                << "rest_pot: " << rest_pot << ", "
                << "tau_ref: " << tau_ref << ", "
                << "noise: " << noise << ", "
                << "tau_adapt: " << tau_adapt << ", "
                << "kd: " << kd << ", "
                << "vK: " << vK << ", "
                << "adapt_amp: " << adapt_amp << ", "
                << "gKCa: " << gKCa << Self::end;
    }

    double tau_m;
    double rest_pot;
    double tau_ref;
    double noise;
    double tau_adapt;
    double kd;
    double vK;
    double adapt_amp;
    double gKCa;
};


/*@GENERATE_PROTO@*/
struct AdaptIntegrateAndFireState : public Serializable<Protos::AdaptIntegrateAndFireState>  {
    AdaptIntegrateAndFireState()
    : ref_time(0.0)
    , Ca(0.0)
    {}

    void serial_process() {
        begin() << "ref_time: " << ref_time << ", "
                << "Ca: "       << Ca << Self::end;
    }
    double ref_time;
    double Ca;
};


class AdaptIntegrateAndFire : public SpikeNeuron<AdaptIntegrateAndFireC, AdaptIntegrateAndFireState> {
public:
    const string name() const {
        return "AdaptIntegrateAndFire";
    }

    void reset() {
        firingProbability() = 0.0;
        membrane() = c.rest_pot;
        s.ref_time = 0.0;
        s.Ca = 0.0;
    }

    void calculateDynamics(const Time& t, const double &Iinput, const double &Isyn) {
        if(s.ref_time < 0.001) {
            double Ia = c.gKCa * (s.Ca/(s.Ca + c.kd)) * (membrane() - c.vK);
            stat.add("Ia", Ia);

            membrane() += t.dt * (
                - membrane()
                + c.noise * getNorm()
                + Iinput
                + Isyn
                - Ia
                ) / c.tau_m;
            firingProbability() = act_f.ifc().prob(membrane());


            if(getUnif() < firingProbability()) {
                setFired(true);
                membrane() = c.rest_pot;
                s.ref_time = c.tau_ref;
                s.Ca += c.adapt_amp;
            }
        } else {
            s.ref_time -= t.dt;
        }
        s.Ca += t.dt * ( - s.Ca/c.tau_adapt );
        stat.add("u", membrane());
        stat.add("Ca", s.Ca);
    }

};

}
