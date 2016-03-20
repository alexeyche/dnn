#pragma once

#include "receptive_field.h"

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/gauss.pb.h>

namespace NDnn {

    struct TGaussReceptiveFieldConst: public IProtoSerial<NDnnProto::TGaussReceptiveFieldConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kGaussReceptiveFieldFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(Sigma);
            serial(Gain);
            serial(LowLevel);
            serial(HighLevel);
        }

        double Sigma = 0.1;
        double Gain = 1.0;
        double LowLevel = 0.0;
        double HighLevel = 1.0;
    };


    class TGaussReceptiveField : public TReceptiveField<TGaussReceptiveFieldConst> {
    public:
        void Init(const TNeuronSpaceInfo& info) {
            Center = c.LowLevel + (c.HighLevel - c.LowLevel) * static_cast<double>(info.LocalId)/info.LayerSize;
        }

        double CalculateResponse(double I) {
            return c.Gain * exp( - (I - Center)*(I - Center) / (2.0 *c.Sigma*c.Sigma));
        }

    private:
        double Center;
    };



}
