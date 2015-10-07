#pragma once

#include "input.h"

#include <dnn/protos/input_time_series.pb.h>
#include <dnn/io/stream.h>
#include <dnn/util/log/log.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct InputTimeSeriesC : public Serializable<Protos::InputTimeSeriesC> {
    InputTimeSeriesC() : dt(1.0) {}

    double dt;

    void serial_process() {
        begin() << "dt: " << dt << Self::end;
    }
};

/*@GENERATE_PROTO@*/
struct InputTimeSeriesState : public Serializable<Protos::InputTimeSeriesState> {
    InputTimeSeriesState() : index(0), __t(0) {}
    size_t index;
    
    double __t;
    
    void serial_process() {
        begin() << "index: " << index << Self::end;
    }
};


class InputTimeSeries : public Input<InputTimeSeriesC, InputTimeSeriesState> {
public:
    typedef Input<InputTimeSeriesC, InputTimeSeriesState> Parent;

    const string name() const {
        return "InputTimeSeries";
    }

	const double& getValue(const Time &t) {
        s.__t += c.dt;
        if(fmod(s.__t, t.dt) > 0.0001) return InputBase::def_value;
        assert(seq->size() > s.index);
        return seq->at(s.index++);
	}

    void setAsInput(Ptr<SerializableBase> b) {
        data_src = b;
        reset();
    }

    void reset() {
        s.index = 0;
        Ptr<TimeSeries> ts = data_src.as<TimeSeries>();
        if(fabs(ts->getTimeDelta() - c.dt) > 1e-03) {
            ts->changeTimeDelta(c.dt);
        }
        size_t id = localId();
        if(ts->dim() == 1) {
            seq.set( &ts.ref().data[0].values );
        } else {
            if(id>=ts->dim()) {
                throw dnnException() << "Got input spike sequence less than neuron count\n";
            }
            seq.set( &ts.ref().data[id].values );
        }
    }

    double getSimDuration() {
        if(seq.isSet()) {
            return seq->size()/c.dt;
        }
        return 0.0;
    }

private:
    Ptr<SerializableBase> data_src;
    Ptr<vector<double>> seq;
};



}
