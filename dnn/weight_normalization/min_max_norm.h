#pragma once

#include "weight_normalization.h"

#include <dnn/protos/min_max_norm.pb.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {

	struct TMinMaxNormConst: public IProtoSerial<NDnnProto::TMinMaxNormConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kMinMaxNormFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	        serial(MinWeight);
	        serial(MaxWeight);
	    }

	    double MinWeight = 0.0;
	    double MaxWeight = 1.0;
	};

	struct TMinMaxNormState: public IProtoSerial<NDnnProto::TMinMaxNormState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kMinMaxNormStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
		}
	};

	template <typename TNeuron>
	class TMinMaxNorm: public TWeightNormalization<TMinMaxNormConst, TMinMaxNormState, TNeuron> {
	public:
		using TPar = TWeightNormalization<TMinMaxNormConst, TMinMaxNormState, TNeuron>;

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

} // namespace NDnn
