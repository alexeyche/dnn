#pragma once

#include "activation.h"

#include <dnn/protos/log_exp.pb.h>
#include <ground/serial/proto_serial.h>

namespace NDnn {

    struct TLogExpConst: public IProtoSerial<NDnnProto::TLogExpConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kLogExpFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(R0);
            serial(U0);
            serial(Ua);
        }

        double R0 = 1.0;
        double U0 = 0.1;
        double Ua = 1.0;
    };


    class TLogExp: public TActivation<TLogExpConst> {
    public:
        double SpikeProbability(const double& u) {
            const double p = c.R0 * std::log(1.0 + std::exp( (u - c.U0) / c.Ua));

            if(fabs(p)<1e-04) {
                return 1e-04;
            }
            return p;
        }

        double SpikeProbabilityDerivative(const double& u) {
            double ev = std::exp( (u - c.U0) / c.Ua);
            return c.R0 * ev / (c.Ua * (ev + 1.0));
        }
    };

} // namespace NDnn
