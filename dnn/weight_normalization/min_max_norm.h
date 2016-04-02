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

	class TMinMaxNorm: public TWeightNormalization<TMinMaxNormConst, TMinMaxNormState> {
	public:
		double DerivativeModulation(const double& w) const {
			if ((std::abs(w) >= c.MaxWeight) || (std::abs(w) <= c.MinWeight)) {
	    		return 0.0;
	    	}
        	return 1.0;
    	}
	};

} // namespace NDnn
