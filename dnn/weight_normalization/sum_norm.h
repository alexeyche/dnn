#pragma once

#include "weight_normalization.h"

#include <dnn/protos/unit_norm.pb.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {

	struct TSumNormConst: public IProtoSerial<NDnnProto::TSumNormConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSumNormFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	        serial(ExcUnit);
	        serial(InhUnit);
	    }

	    double ExcUnit = 1.0;
	    double InhUnit = 1.0;
	};

	struct TSumNormState: public IProtoSerial<NDnnProto::TSumNormState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSumNormStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
		}
	};

	template <typename TNeuron>
	class TSumNorm: public TWeightNormalization<TSumNormConst, TSumNormState, TNeuron> {
	public:
		using TPar = TWeightNormalization<TSumNormConst, TSumNormState, TNeuron>;
        
		double Derivative(double w, double dw) const {
			if ((w < 0.0) != ((w+dw)<0.0)) {
				return 0.0;
			}
			if ((std::abs(w+dw) > TPar::c.ExcUnit) || (std::abs(w+dw) < 0.0)) {
	    		return 0.0;
	    	}

	    	return dw;
    	}

        void CalculateDynamics(const TTime &t) {
        	// if (std::fmod(t.T, 10000.0) > std::numeric_limits<double>::epsilon()) {
        	// 	return;
        	// }
        	
        	auto& syns = TPar::GetMutSynapses();
	        
	        double excAcc = 0.0;
	        double inhAcc = 0.0;
	        for (const auto& s: syns) {
	        	if (s.Weight() >= 0.0) {
	        		excAcc += s.Weight();
	        	} else {
	        		inhAcc += std::abs(s.Weight());
	        	}
	        }
	        
	        double excDenom = TPar::c.ExcUnit/excAcc;
	        double inhDenom = TPar::c.InhUnit/inhAcc;
	        for (auto& s: syns) {
	        	if (s.Weight() >= 0.0) {
	        		s.MutWeight() = s.Weight()*excDenom;	
	        	} else {
	        		s.MutWeight() = s.Weight()*inhDenom;
	        	}
	        }
        }
	};

} // namespace NDnn
