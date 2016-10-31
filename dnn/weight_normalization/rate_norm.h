#pragma once


#include "weight_normalization.h"

#include <dnn/protos/rate_norm.pb.h>

namespace NDnn {

    struct TRateNormConst : public IProtoSerial<NDnnProto::TRateNormConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kRateNormFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Rmin);
            serial(Rmax);
            serial(TauMean);
            serial(F);
            serial(MinWeight);
            serial(MaxWeight);
        }

        double Rmin = 1.0;
        double Rmax = 10.0;
        double TauMean = 1000.0;
        double F = 0.05;
        double MinWeight = 0.0;
        double MaxWeight = 1.0;
    };

    struct TRateNormState: public IProtoSerial<NDnnProto::TRateNormState>  {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kRateNormStateFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Rate);
            serial(Saturation);
        }

        double Rate = 0.0;
        bool Saturation = false;
    };


    template <typename TNeuron>
    class TRateNorm : public TWeightNormalization<TRateNormConst, TRateNormState, TNeuron> {
    public:
        using TPar = TWeightNormalization<TRateNormConst, TRateNormState, TNeuron>;

        void CalculateDynamics(const TTime& t) {
            if (!TPar::s.Saturation) {
                TPar::s.Saturation = (TGlobalCtx::Inst().GetPastTime() + t.T) > TPar::c.TauMean;
            }
            TPar::s.Rate += t.Dt * (-TPar::s.Rate + static_cast<double>(TPar::Neuron().Fired()))/TPar::c.TauMean;
        }

        double AlterWeight(double w, double f) const {
            if (w >= 0.0) {
                return (1.0 + f) * w;
            } else {
                return (1.0/(1.0 + f)) * w;
            }
        }

        double Derivative(double& w, double dw) const {
            if (!TPar::s.Saturation) {
                return 0.0;
            }

            if ((w < 0.0) != ((w+dw)<0.0)) {
                return 0.0;
            }
            if ((std::abs(w+dw) >= TPar::c.MaxWeight) || (std::abs(w+dw) <= TPar::c.MinWeight)) {
                return 0.0;
            }
            
            double r = TPar::s.Rate*1000.0;
            if (r <= TPar::c.Rmin) {
                double new_w = AlterWeight(w, TPar::c.F);
                if ((w > 0.0) == ((new_w+dw) > 0.0)) {
                    // L_INFO << r << ", +, " << w << " -> " << new_w;
                    w = new_w;
                }
            } else
            if (r >= TPar::c.Rmax) {
                double new_w = AlterWeight(w, -TPar::c.F);
                if ((w > 0.0) == ((new_w+dw) > 0.0)) {
                    // L_INFO << r << ", -, " << w << " -> " << new_w;
                    w = new_w;    
                } 
            }

            return dw;
        }
    };



} // namespace NDnn
