// #pragma once

// #include "weight_normalization.h"

// #include <dnn/protos/soft_bounds.pb.h>
// #include <dnn/protos/config.pb.h>

// namespace NDnn {

// 	struct TSoftBoundsConst: public IProtoSerial<NDnnProto::TSoftBoundsConst> {
// 		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSoftBoundsFieldNumber;

// 	    void SerialProcess(TProtoSerial& serial) {
// 	    	serial(MinWeight);
// 	        serial(MaxWeight);
// 	    }

// 	    double MinWeight = 0.0;
// 	    double MaxWeight = 1.0;
// 	};

// 	struct TSoftBoundsState: public IProtoSerial<NDnnProto::TSoftBoundsState>  {
// 		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSoftBoundsStateFieldNumber;

// 		void SerialProcess(TProtoSerial& serial) {
// 		}
// 	};

// 	class TSoftBounds: public TWeightNormalization<TSoftBoundsConst, TSoftBoundsState> {
// 	public:
// 		double Ltp(double w) const {
// 	        return c.MaxWeight - std::abs(w);
// 	    }

// 	    double Ltd(double w) const {
// 	    	return std::abs(w);
// 	    }
// 	};

// } // namespace NDnn
