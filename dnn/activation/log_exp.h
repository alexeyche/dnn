#pragma once

#include "activation.h"

#include <dnn/protos/log_exp.pb.h>
#include <ground/serial/proto_serial.h>

namespace NDnn {

    struct TLogExpConst: public IProtoSerial<NDnnProto::TLogExpConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kLogExpFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Threshold);
            serial(Slope);
        }

        double Threshold = 0.1;
        double Slope = 1.0;
    };


    class TLogExp: public TActivation<TLogExpConst> {
    public:
        double SpikeProbability(const double& u) const {
            const double p = std::log( (1.0 + std::exp( (u - c.Threshold) / c.Slope)) / (1.0 + std::exp( - c.Threshold / c.Slope)) );
            // if (p < 1e-05) {
            //     return 1e-05;
            // }
            return p;
        }

        double SpikeProbabilityDerivative(const double& u) const {
            const double exp_x1 = std::exp((u - c.Threshold) / c.Slope);
            const double exp_x2 = std::exp(- c.Threshold / c.Slope);
            
            return (1.0/c.Slope) * exp_x1/(exp_x1 + exp_x2 + 1.0);
        }
    };

} // namespace NDnn
