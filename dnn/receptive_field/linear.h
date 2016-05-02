#pragma once

#include "receptive_field.h"

#include <ground/serial/proto_serial.h>
#include <dnn/protos/linear.pb.h>

namespace NDnn {

    struct TLinearReceptiveFieldConst: public IProtoSerial<NDnnProto::TLinearReceptiveFieldConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kLinearReceptiveFieldFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(TauRef);
            serial(TauRc);
            serial(LowLevel);
            serial(HighLevel);
            serial(LowRate);
            serial(HighRate);
        }

        double TauRef = 2.0;
        double TauRc = 20.0;
        double LowLevel = 0.0;
        double HighLevel = 1.0;
        double LowRate = 2.0;
        double HighRate = 30.0;
    };


    class TLinearReceptiveField : public TReceptiveField<TLinearReceptiveFieldConst> {
    public:
        void Init(const TNeuronSpaceInfo& info, TRandEngine& rand) {
            double intercept = c.LowLevel + (c.HighLevel - c.LowLevel) * rand.GetUnif();
            double rate = c.LowRate + (c.HighRate - c.LowRate) * rand.GetUnif();
            double z = 1.0 / (1.0 - exp((c.TauRef/1000.0 - 1.0/rate)/(c.TauRc/1000.0)));
            Gain = (1.0 - z) / (intercept - 1.0);
            Bias = 1.0 - Gain*intercept;
            Encoder = rand.GetUnif() > 0.5 ? 1.0 : -1.0;
        }

        double CalculateResponse(double I) {
            return I * Encoder * Gain + Bias;
        }

    private:
        double Gain;
        double Bias;
        double Encoder;
    };



}
