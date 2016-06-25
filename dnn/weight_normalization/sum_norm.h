#pragma once

#include "weight_normalization.h"

#include <dnn/protos/unit_norm.pb.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {

	struct TLayerNorm: public IProtoSerial<NDnnProto::TLayerNorm> {
		
		void SerialProcess(TProtoSerial& serial) {
			serial(Id);
	        serial(ExcUnit);
	        serial(InhUnit);
	    }

		ui32 Id = 0;
		double ExcUnit = 1.0;
	    double InhUnit = 1.0;
	};

	struct TSumNormConst: public IProtoSerial<NDnnProto::TSumNormConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSumNormFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	        serial(Layer);
	    }

	    TVector<TLayerNorm> Layer;
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

        void Reset() {
        	ENSURE(TPar::c.Layer.size() > 0, "Need to point per layer information for SumNorm");
        	TVector<TLayerNorm> layer;
        	layer.resize(TGlobalCtx::Inst().GetLayerSize(), *TPar::c.Layer.begin());
        	for (const auto& layerNorm: TPar::c.Layer) {
        		layer[layerNorm.Id] = layerNorm;
        		MaxUnitNorm = std::max(MaxUnitNorm, layer[layerNorm.Id].ExcUnit);
        	}
        	TPar::c.Layer = layer;
    	}

		double Derivative(double w, double dw) const {
			if ((w < 0.0) != ((w+dw)<0.0)) {
				return 0.0;
			}
			if ((std::abs(w+dw) > MaxUnitNorm) || (std::abs(w+dw) < 0.0)) {
	    		return 0.0;
	    	}

	    	return dw;
    	}
    	

        void CalculateDynamics(const TTime &t) {
        	// if (std::fmod(t.T, 10000.0) > std::numeric_limits<double>::epsilon()) {
        	// 	return;
        	// }
        	
        	auto& syns = TPar::GetMutSynapses();
	        TVector<double> excAcc(TPar::c.Layer.size());
	        TVector<double> inhAcc(TPar::c.Layer.size());
	        
			TVector<double> excAccOld(TPar::c.Layer.size());
	        TVector<double> inhAccOld(TPar::c.Layer.size());
	        

	        for (const auto& s: syns) {
	        	const auto& layerId = TGlobalCtx::Inst().GetNeuronLayerId(s.IdPre());
	        	if (s.Weight() >= 0.0) {
	        		excAcc[layerId] += s.Weight();
	        	} else {
	        		inhAcc[layerId] += std::abs(s.Weight());
	        	}
	        }
	        
	        for (ui32 layerId=0; layerId < TPar::c.Layer.size(); ++layerId) {
	        	excAccOld[layerId] = excAcc[layerId];
	        	excAcc[layerId] = TPar::c.Layer[layerId].ExcUnit/excAcc[layerId];
	        	inhAcc[layerId] = TPar::c.Layer[layerId].InhUnit/inhAcc[layerId];
	        }
	        
	        for (auto& s: syns) {
	        	const auto& layerId = TGlobalCtx::Inst().GetNeuronLayerId(s.IdPre());
	        	
	        	if (s.Weight() >= 0.0) {
	        		// if (layerId == 0) {
		        	// 	L_INFO << excAcc[layerId];
		        	// }
	        		s.MutWeight() = s.Weight()*excAcc[layerId];	
	        	} else {
	        		// if (layerId == 0) {
		        	// 	L_INFO << inhAcc[layerId];
		        	// }
	        		s.MutWeight() = s.Weight()*inhAcc[layerId];
	        	}
	        	// if ((TPar::SpaceInfo().GlobalId == 345) && (s.Weight() > 2.0)) {
	        	// 	L_INFO << "BAD: " << "T: " << t.T << " " << excAccOld[layerId] << " " << excAcc[layerId] << " " << TPar::c.Layer[layerId].ExcUnit;
	        	// }
	        }
        }

        TVector<TLayerNorm>& GetMutLayerNorm() {
        	return TPar::c.Layer;
        }

    private:
    	double MaxUnitNorm;
	};

} // namespace NDnn
