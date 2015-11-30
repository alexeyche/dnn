#pragma once

#include "input.h"

#include <dnn/protos/input_time_series.pb.h>
#include <dnn/io/stream.h>
#include <dnn/util/log/log.h>
#include <dnn/sim/global_ctx.h>

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
    InputTimeSeriesState() : t(0) {}

    double t;

    void serial_process() {
        begin() << "t: " << t << Self::end;
    }
};


class InputTimeSeries : public Input<InputTimeSeriesC, InputTimeSeriesState> {
public:
    typedef Input<InputTimeSeriesC, InputTimeSeriesState> Parent;


	const double getValue(const Time &t) {
        double index = floor(s.t);
        if(s.t-index < c.dt) {
            s.t += c.dt;
            assert(seq->size() > index);
            return seq->at(index);
        }
        s.t += c.dt;
        return InputBase::def_value;
	}

    void setAsInput(Ptr<SerializableBase> b) {
        data_src = b;
        reset();
    }

    void init() {
        if(!seq.isSet() || seq->size() == 0) return;
        GlobalCtx::inst().setSimDuration(seq->size()/c.dt);
    }

    void reset() {
        s.t = 0;
        Ptr<TimeSeries> ts = data_src.as<TimeSeries>();
        if(fabs(ts->getTimeDelta() - c.dt) > 1e-03) {
            L_DEBUG << "Found input time series with time delta " << ts->getTimeDelta() << ". Changing it to " << c.dt;
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


private:
    Ptr<SerializableBase> data_src;
    Ptr<vector<double>> seq;
};



}
