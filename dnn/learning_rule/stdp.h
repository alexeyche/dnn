#pragma once

#include "learning_rule.h"

#include <dnn/protos/stdp.pb.h>
#include <dnn/protos/config.pb.h>
#include <ground/act_vector.h>

namespace NDnn {

	struct TStdpConst: public IProtoSerial<NDnnProto::TStdpConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kStdpFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	        serial(TauPlus);
	        serial(TauMinus);
	        serial(Aplus);
	        serial(Aminus);
	        serial(LearningRate);
	    }

	    double TauPlus = 30.0;
	    double TauMinus = 50.0;
	    double Aplus = 1.0;
	    double Aminus = 1.0;
	    double LearningRate = 0.01;
	};

	struct TStdpState: public IProtoSerial<NDnnProto::TStdpState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kStdpStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
			serial(Y);
			serial(X);
		}

		double Y = 0.0;
    	TActVector<double> X;
	};

	template <typename TNeuron, typename TWeightNormalizationType>
	class TStdp: public TLearningRule<TStdpConst, TStdpState, TNeuron, TWeightNormalizationType> {
	public:
		using TPar = TLearningRule<TStdpConst, TStdpState, TNeuron, TWeightNormalizationType>;

		void Reset() {
			TPar::s.Y = 0.0;
			TPar::s.X.resize(TPar::GetSynapses().size());
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
	        TPar::s.X[sp.SynapseId] += 1.0;
    	}

    	void CalculateDynamics(const TTime& t) {
    		auto& syns = TPar::GetMutSynapses();
    		const auto& norm = TPar::Norm();
    		const auto& neuron = TPar::Neuron();
    		
    		if (neuron.Fired()) {
    			TPar::s.Y += 1.0;
    		}
	        
			auto synIdIt = TPar::s.X.abegin();
		    while (synIdIt != TPar::s.X.aend()) {
		    	if (std::fabs(TPar::s.X[synIdIt]) < 0.0001) {
                	TPar::s.X.SetInactive(synIdIt);
                } else {
	                const ui32& synapseId = *synIdIt;
	                auto& syn = syns.Get(synapseId);
	                double& w = syn.MutWeight();

	                double dw = TPar::c.LearningRate * (
	                    TPar::c.Aplus  * TPar::s.X[synIdIt] * neuron.Fired() * norm.Ltp(w) -
	                    TPar::c.Aminus * TPar::s.Y * syn.Fired() * norm.Ltd(w)
	                );
	                
	                w += norm.Derivative(w, dw);

	                TPar::s.X[synIdIt] += - t.Dt * TPar::s.X[synIdIt]/TPar::c.TauPlus;
                	++synIdIt;
                }
            }

            TPar::s.Y += - t.Dt * TPar::s.Y/TPar::c.TauMinus;
    	}
	};

} // namespace NDnn
