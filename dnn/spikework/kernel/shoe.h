#pragma once

#include "kernel.h"

#include <dnn/spikework/protos/spikework_config.pb.h>

namespace NDnn {

    struct TShoeOptions: IProtoSerial<NDnnProto::TShoeOptions> {
        static const auto ProtoFieldNumber = NDnnProto::TKernelConfig::kShoeFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Sigma);
            serial(Kernel);
        }

        double Sigma = 0.1;
        NDnnProto::TKernelConfig Kernel;
    };


    class TShoeKernel: public IKernel {
    public:
        double PointSimilarity(const TVector<double>& x, const TVector<double>& y) const override final;

        void SerialProcess(TProtoSerial& serial) override;
        
    private:
        TShoeOptions Options;
        SPtr<IKernel> Kernel;
    };



} // namespace NDnn