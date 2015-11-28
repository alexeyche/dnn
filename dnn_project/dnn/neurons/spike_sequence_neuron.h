#pragma once


#include "spike_neuron.h"
#include <dnn/protos/spike_sequence_neuron.pb.h>
#include <dnn/io/serialize.h>
#include <dnn/sim/global_ctx.h>

namespace dnn {


/*@GENERATE_PROTO@*/
struct SpikeSequenceNeuronC : public Serializable<Protos::SpikeSequenceNeuronC> {
    SpikeSequenceNeuronC()
    : dt(1.0)
    {}

    void serial_process() {
        begin() << "dt: " << dt << Self::end;
        if(fabs(dt) < 1e-05) {
            throw dnnException() << "Got very little dt in SpikeSequenceNeuron\n";
        }
    }

    double dt;
};


/*@GENERATE_PROTO@*/
struct SpikeSequenceNeuronState : public Serializable<Protos::SpikeSequenceNeuronState>  {
    SpikeSequenceNeuronState()
    : index(0)
    {}

    void serial_process() {
        begin() << "index: " << index << Self::end;
    }
    size_t index;
};


class SpikeSequenceNeuron : public SpikeNeuron<SpikeSequenceNeuronC, SpikeSequenceNeuronState> {
public:

    void init() {
        if(!seq.isSet() || seq->size() == 0) return;

        if(input.isSet()) {
            throw dnnException() << "Got current inputs in SpikeSequenceNeuron. Config is errors prone\n";
        }
        if(lrule.isSet()) {
            throw dnnException() << "Got learning rule in SpikeSequenceNeuron. Config is errors prone\n";
        }
        if(act_f.isSet()) {
            throw dnnException() << "Got activation function in SpikeSequenceNeuron. Config is errors prone\n";
        }

        GlobalCtx::inst().setSimDuration(seq->back());
    }

    void reset() {
        s.index = 0;
    }

    void calculateDynamics(const Time& t, const double &Iinput, const double &Isyn) {
        assert(seq.isSet());
        if(s.index>=seq->size()) return;
        double spike_time = seq->at(s.index);
        if((spike_time>=t.t)&&(spike_time<(t.t+t.dt))) {
            s.index++;
            setFired(true);
        }
    }

    void setAsInput(Ptr<SerializableBase> b) {
        Ptr<SpikesList> sl = b.as<SpikesList>();
        if(fabs(sl->getTimeDelta() - c.dt) > 1e-03) {
            sl->changeTimeDelta(c.dt);
        }
        size_t id = localId();
        if(id>=sl->size()) {
            throw dnnException() << "Got input spike sequence less than neuron count\n";
        }
        seq.set( &sl->seq[id].values );
    }


private:
    Ptr< vector<double> > seq;
};

}
