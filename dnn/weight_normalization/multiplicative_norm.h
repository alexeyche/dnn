#pragma once


#include "weight_normalization.h"

#include <dnn/protos/multiplicative_norm.pb.h>

namespace NDnn {

    struct TMultiplicativeNormConst : public IProtoSerial<NDnnProto::TMultiplicativeNormConst> {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kMultiplicativeNormFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
            serial(Unit);
        }

        double Unit = 2.0;
    };

    struct TMultiplicativeNormState: public IProtoSerial<NDnnProto::TMultiplicativeNormState>  {
        static const auto ProtoFieldNumber = NDnnProto::TLayer::kMultiplicativeNormStateFieldNumber;

        void SerialProcess(TProtoSerial& serial) {
        }
    };

    template <typename TNeuron>
    class TMultiplicativeNorm : public TWeightNormalization<TMultiplicativeNormConst, TMultiplicativeNormState, TNeuron> {
    public:
        using TPar = TWeightNormalization<TMultiplicativeNormConst, TMultiplicativeNormState, TNeuron>;

        double Derivative(double w, double dw) const {
            double new_dw = dw - w * (w * dw);
            
            if ((w < 0.0) != ((w+new_dw)<0.0)) {
                return 0.0;
            }
            if ((std::abs(w+new_dw) >= TPar::c.Unit) || (std::abs(w+new_dw) <= TPar::c.Unit)) {
                return 0.0;
            }
            return new_dw;
        }
    };



} // namespace NDnn
