#pragma once

#include "kernel.h"

#include <dnn/spikework/protos/spikework_config.pb.h>

namespace NDnn {

    struct TRbfDotOptions: IProtoSerial<NDnnProto::TRbfDotOptions> {
        static const auto ProtoFieldNumber = NDnnProto::TKernelConfig::kRbfDotFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Sigma);
        }

        double Sigma = 0.1;
    };


    class TRbfDotKernel: public IKernel {
    public:
        double PointSimilarity(const TVector<double>& x, const TVector<double>& y) const override final;

        void SerialProcess(TProtoSerial& serial) override {
            serial(Options, NDnnProto::TKernelConfig::kRbfDotFieldNumber);
        }
        
    private:
        TRbfDotOptions Options;
    };



} // namespace NDnn