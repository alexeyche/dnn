#pragma once

#include "activation.h"

#include <dnn/protos/sigmoid.pb.h>
#include <ground/fastapprox/fastexp.h>
#include <ground/fastapprox/fastlog.h>
#include <ground/serial/proto_serial.h>

namespace NDnn {

    struct TSigmoidConst: public IProtoSerial<NDnnProto::TSigmoidConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kSigmoidFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Threshold);
            serial(Slope);
        }

        double Threshold = 0.2;
        double Slope = 1.0;
    };


    class TSigmoid: public TActivation<TSigmoidConst> {
    public:
        double SpikeProbability(const double& u) const {
            const double input = (u - c.Threshold)/c.Slope;
            const double p = 1.0/(1.0+std::exp(-input));
            return p;
        }

        double SpikeProbabilityDerivative(const double& u) const {
            double p = SpikeProbability(u);
            return p*(1.0-p);
        }
    };

} // namespace NDnn
