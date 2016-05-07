#pragma once

#include "weight_normalization.h"

#include <dnn/protos/nnmf_homeostasis.pb.h>

namespace NDnn {

    struct TNNMFHomeostatisConst : public IProtoSerial<NDnnProto::TNNMFHomeostatisConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kNNMFHomeostatisFieldNumber;
        TNNMFHomeostatisConst() {
            __TargetRateMs = TargetRate/1000.0;
        }  

        void SerialProcess(TProtoSerial& serial) override final {
            serial(TargetRate);
            serial(TauMean);
            serial(Alpha);
            serial(Beta);
            serial(Gamma);
            serial(MinWeight);
            serial(MaxWeight);

            __TargetRateMs = TargetRate/1000.0;
        }

        double TargetRate = 5.0;
        double TauMean = 10000;
        double Alpha = 0.1;
        double Beta = 1.0;
        double Gamma = 10.0;
        double MinWeight = 0.0;
        double MaxWeight = 1.0;
        double __TargetRateMs;
    };


    struct TNNMFHomeostatisState : public IProtoSerial<NDnnProto::TNNMFHomeostatisState>  {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kNNMFHomeostatisStateFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(Pmean);
            serial(Saturation);
        }
        double Pmean = 0.0;
        bool Saturation = false;
    };

    template <typename TNeuron>
    class TNNMFHomeostatis : public TWeightNormalization<TNNMFHomeostatisConst, TNNMFHomeostatisState, TNeuron> {
    public:
        using TPar = TWeightNormalization<TNNMFHomeostatisConst, TNNMFHomeostatisState, TNeuron>;

        void CalculateDynamics(const TTime& t) {
            if (!TPar::s.Saturation) {
                TPar::s.Saturation = (TGlobalCtx::Inst().GetPastTime() + t.T) > TPar::c.TauMean;
            }
            TPar::s.Pmean += t.Dt * (-TPar::s.Pmean + static_cast<double>(TPar::Neuron().Fired()))/TPar::c.TauMean;
        }

        double Derivative(double w, double dw) const {
            if (!TPar::s.Saturation) {
                return 0.0;
            }
            
            double pratio = 1.0 - TPar::s.Pmean/TPar::c.__TargetRateMs;
            double K = TPar::s.Pmean / (TPar::c.TauMean * (1.0 + std::abs(pratio) * TPar::c.Gamma));
            return K * (TPar::c.Alpha * w * pratio + TPar::c.Beta * dw);
        }
    };



}
