#pragma once

#include "kernel.h"

#include <dnn/spikework/protos/spikework_config.pb.h>

namespace NDnn {

    struct TDotOptions: IProtoSerial<NDnnProto::TDotOptions> {
        static const auto ProtoFieldNumber = NDnnProto::TKernelConfig::kDotFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
        }
    };


    class TDotKernel: public IKernel {
    public:
        double PointSimilarity(const TVector<double>& x, const TVector<double>& y) const override final;

        void SerialProcess(TProtoSerial& serial) override {}
    };



} // namespace NDnn