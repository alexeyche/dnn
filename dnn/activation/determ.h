#pragma once

#include "activation.h"

#include <ground/serial/proto_serial.h>
#include <dnn/protos/determ.pb.h>

namespace NDnn {

    struct TDetermConst: public IProtoSerial<NDnnProto::TDetermConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kDetermFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(Threshold);
        }

        double Threshold = 1.0;
    };


    class TDeterm : public TActivation<TDetermConst> {
    public:
        double SpikeProbability(const double& u) const {
            if(u >= c.Threshold) {
                return 1.0;
            }
            return 0.0;
        }

        double SpikeProbabilityDerivative(const double &u) const {
            return 1.0;
        }
    };


} // namespace NDnn