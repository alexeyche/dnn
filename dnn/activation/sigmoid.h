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


        double Threshold = 0.15;
        double Slope = 50.0;
    };


    class TSigmoid: public TActivation<TSigmoidConst> {
    public:
        double SpikeProbability(const double& u) {
            double p = 1.0/(1.0+exp( - c.Slope * (u - c.Threshold) ));
            if(fabs(p)<1e-04) {
                return 1e-04;
            }
            return p;
        }

        double SpikeProbabilityDerivative(const double& u) {
            double p = SpikeProbability(u);
            return p*(1.0-p);
        }
    };

} // namespace NDnn
