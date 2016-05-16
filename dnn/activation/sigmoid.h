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
            serial(A);
            serial(B);
        }

        double A = 20.0;
        double B = -4.0;
    };


    class TSigmoid: public TActivation<TSigmoidConst> {
    public:
        double SpikeProbability(const double& u) {
            const double input = c.A*u + c.B;
            const double p = 1.0/(1.0+std::exp(-input));
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
