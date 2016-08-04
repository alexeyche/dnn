#pragma once


#include "weight_normalization.h"

#include <dnn/protos/log_norm.pb.h>

namespace NDnn {

    // Stability versus Neuronal Specialization for STDP: Long-Tail Weight Distributions Solve the Dilemma

    struct TLogNormConst : public IProtoSerial<NDnnProto::TLogNormConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kLogNormFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(MeanWeight);
            serial(Alpha);
            serial(Beta);
        }

        double MeanWeight = 0.05;
        double Alpha = 5.0;
        double Beta = 50.0;
    };

    struct TLogNormState: public IProtoSerial<NDnnProto::TLogNormState>  {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kLogNormStateFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
        }
    };

    template <typename TNeuron>
    class TLogNorm : public TWeightNormalization<TLogNormConst, TLogNormState, TNeuron> {
    public:
        using TPar = TWeightNormalization<TLogNormConst, TLogNormState, TNeuron>;

        double Ltp(double w) const {
            return std::exp(-std::abs(w)/(TPar::c.MeanWeight * TPar::c.Beta));
        }

        double Ltd(double w) const {
            double prop = std::abs(w)/TPar::c.MeanWeight;
            if (std::abs(w) <= TPar::c.MeanWeight) {
                return prop;
            }
            return 1.0 + std::log(1.0 + TPar::c.Alpha * (prop - 1.0))/TPar::c.Alpha;
        }

        double Derivative(double w, double dw) const {
            if ((w < 0.0) != ((w+dw)<0.0)) {
                return 0.0;
            }
            return dw;
        }
    };



} // namespace NDnn
