#pragma once

#include "kernel.h"

#include <dnn/spikework/protos/spikework_config.pb.h>

namespace NDnn {

    struct TAnovaDotOptions: IProtoSerial<NDnnProto::TAnovaDotOptions> {
        static const auto ProtoFieldNumber = NDnnProto::TKernelConfig::kAnovaDotFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Sigma);
            serial(Power);
        }

        double Sigma = 0.1;
        double Power = 1.0;
    };


    class TAnovaDotKernel: public IKernel {
    public:
        double PointSimilarity(const TVector<double>& x, const TVector<double>& y) const override final;

        void SerialProcess(TProtoSerial& serial) override {
            serial(Options, NDnnProto::TKernelConfig::kAnovaDotFieldNumber);
        }
        
    private:
        TAnovaDotOptions Options;
    };



} // namespace NDnn