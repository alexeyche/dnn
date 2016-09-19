#pragma once

#include "activation.h"

#include <dnn/protos/exp.pb.h>
#include <ground/serial/proto_serial.h>

namespace NDnn {

    struct TExpConst: public IProtoSerial<NDnnProto::TExpConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kExpFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Threshold);
            serial(Slope);
            serial(Amplitude);
        }

        double Threshold = 0.1;
        double Slope = 1.0;
        double Amplitude = 1.0;
    };


    class TExp: public TActivation<TExpConst> {
    public:
        double SpikeProbability(const double& u) const {
            const double p = c.Amplitude * std::exp((u - c.Threshold)/c.Slope);
            // L_INFO << c.Slope << " " << u - c.Threshold << " " << (u - c.Threshold)/c.Slope << " " << p;
            if (p < 1e-06) {
                return 1e-06;
            }
            return p;
        }

        double SpikeProbabilityDerivative(const double& u) const {
            const double p = SpikeProbability(u);
            return p/c.Slope;
        }
    };

} // namespace NDnn
