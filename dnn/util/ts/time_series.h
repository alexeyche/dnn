#pragma once

#include "generic.h"

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/time_series.pb.h>

namespace NDnn {

    struct TTimeSeriesData : public IProtoSerial<NDnnProto::TTimeSeriesData> {
        using TElement = double;

        void SerialProcess(TProtoSerial& serial) {
            serial(Values);
        }

        TVector<double> Values;
    };


    struct TTimeSeries: public IProtoSerial<NDnnProto::TTimeSeries>, public TTimeSeriesGeneric<TTimeSeriesData> {

        void SerialProcess(TProtoSerial& serial) {
            serial(Info);
            serial(Data);
        }
    };

}

