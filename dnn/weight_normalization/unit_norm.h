#pragma once

#include "weight_normalization.h"

#include <dnn/protos/unit_norm.pb.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {

	struct TUnitNormConst: public IProtoSerial<NDnnProto::TUnitNormConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kUnitNormFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	        serial(Unit);
	        serial(Power);
	    }

	    double Unit = 1.0;
	    double Power = 2.0;
	};

	struct TUnitNormState: public IProtoSerial<NDnnProto::TUnitNormState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kUnitNormStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
		}
	};

	template <typename TNeuron>
	class TUnitNorm: public TWeightNormalization<TUnitNormConst, TUnitNormState, TNeuron> {
	public:
		using TPar = TWeightNormalization<TUnitNormConst, TUnitNormState, TNeuron>;
        
		double Derivative(double w, double dw) const {
			if ((w < 0.0) != ((w+dw)<0.0)) {
				return 0.0;
			}
			if ((std::abs(w+dw) > TPar::c.Unit) || (std::abs(w+dw) < 0.0)) {
	    		return 0.0;
	    	}
        	return dw;
    	}

        void CalculateDynamics(const TTime &t) {
        	auto& syns = TPar::GetMutSynapses();
	        
	        double denominator = 0.0;
	        for (const auto& s: syns) {
	            denominator += std::pow(s.Weight(), TPar::c.Power);
	        }
	        
	        double mod = TPar::c.Unit/std::pow(denominator, 1.0/TPar::c.Power);
	        for (auto& s: syns) {
	            s.MutWeight() = s.Weight()*mod;
	        }
        }
	};

} // namespace NDnn
