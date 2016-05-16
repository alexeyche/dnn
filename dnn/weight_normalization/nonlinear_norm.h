#pragma once

#include "weight_normalization.h"

#include <dnn/protos/nonlinear_norm.pb.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {

	struct TNonLinearNormConst: public IProtoSerial<NDnnProto::TNonLinearNormConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kNonLinearNormFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	    	serial(DepressionFactor);
	    	serial(Mu);
	        serial(MinWeight);
	        serial(MaxWeight);
	    }

	    double DepressionFactor = 1.0;
	    double Mu = 0.5;
	    double MinWeight = 0.0;
	    double MaxWeight = 1.0;
	};

	struct TNonLinearNormState: public IProtoSerial<NDnnProto::TNonLinearNormState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kNonLinearNormStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
		}
	};

	class TNonLinearNorm: public TWeightNormalization<TNonLinearNormConst, TNonLinearNormState> {
	public:
		double Ltp(double w) const {
	        return std::pow(c.MaxWeight - std::abs(w), c.Mu);
	    }

	    double Ltd(double w) const {
	    	return c.DepressionFactor * std::pow(std::abs(w), c.Mu);
	    }

	    double Derivative(double w, double dw) const {
            if ((w < 0.0) != ((w+dw)<0.0)) {
                return 0.0;
            }
            return dw;
        }
	};

} // namespace NDnn
