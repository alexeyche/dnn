#pragma once

#include "generic.h"

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/time_series.pb.h>

namespace NDnn {

    struct TTimeSeriesComplexData : public IProtoSerial<NDnnProto::TTimeSeriesData> {
        using TElement = TComplex;

        void SerialProcess(TProtoSerial& serial) {
            serial(Values);
        }

        TVector<TComplex> Values;
    };


    struct TTimeSeriesComplex: public IProtoSerial<NDnnProto::TTimeSeriesComplex>, public TTimeSeriesGeneric<TTimeSeriesComplexData> {

        void SerialProcess(TProtoSerial& serial) {
            serial(Info);
            serial(Data);
        }
    };

} // namespace NDnn
