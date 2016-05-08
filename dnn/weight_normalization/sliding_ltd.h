#pragma once

#include "weight_normalization.h"

#include <dnn/protos/sliding_ltd.pb.h>

namespace NDnn {

    struct TSlidingLtdConst : public IProtoSerial<NDnnProto::TSlidingLtdConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kSlidingLtdFieldNumber;

        TSlidingLtdConst() {
            __TargetRate = 1.0/std::pow(TargetRate, Power);
        }

        void SerialProcess(TProtoSerial& serial) override final {
            serial(Power);
            serial(Modulation);
            serial(TargetRate);
            serial(TauMean);
            serial(MinWeight);
            serial(MaxWeight);

            __TargetRate = 1.0/std::pow(TargetRate, Power);
        }

        double Power = 3.0;
        double Modulation = 1.0;
        double TargetRate = 5.0;
        double TauMean = 10000;
        double MinWeight = 0.0;
        double MaxWeight = 1.0;
        double __TargetRate;
    };


    struct TSlidingLtdState : public IProtoSerial<NDnnProto::TSlidingLtdState>  {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kSlidingLtdStateFieldNumber;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(Pmean);
            serial(Saturation);
        }
        double Pmean = 0.0;
        bool Saturation = false;
    };

    template <typename TNeuron>
    class TSlidingLtd : public TWeightNormalization<TSlidingLtdConst, TSlidingLtdState, TNeuron> {
    public:
        using TPar = TWeightNormalization<TSlidingLtdConst, TSlidingLtdState, TNeuron>;

        double Ltd(double w) const {
            if(!TPar::s.Saturation) {
                return 0.0;
            }
            return TPar::c.Modulation * std::pow(1000.0*TPar::s.Pmean, TPar::c.Power) * TPar::c.__TargetRate;
        }

        double Ltp(double w) const {
            if (!TPar::s.Saturation) {
                return 0.0;
            }
            return 1.0;
        }

        void CalculateDynamics(const TTime &t) {
            if (!TPar::s.Saturation) {
                TPar::s.Saturation = (TGlobalCtx::Inst().GetPastTime() + t.T) > TPar::c.TauMean;
            }
            TPar::s.Pmean += t.Dt * (-TPar::s.Pmean + static_cast<double>(TPar::Neuron().Fired()))/TPar::c.TauMean;
        }

        double Derivative(double w, double dw) const {
            if ((w < 0.0) != ((w+dw)<0.0)) {
                return 0.0;
            }
            if ((std::abs(w+dw) >= TPar::c.MaxWeight) || (std::abs(w+dw) <= TPar::c.MinWeight)) {
                return 0.0;
            }
            return dw;
        }
    };



}
