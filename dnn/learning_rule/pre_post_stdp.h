#pragma once

#include "learning_rule.h"

#include <dnn/protos/pre_post_stdp.pb.h>
#include <dnn/protos/config.pb.h>
#include <dnn/util/act_vector.h>

namespace NDnn {

	struct TPrePostStdpConst: public IProtoSerial<NDnnProto::TPrePostStdpConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kPrePostStdpFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	        serial(TauYplus);
	        serial(TauYminus);
	        serial(TauXplus);
	        serial(Cplus);
	        serial(Cminus);
	        serial(Dplus);
	        serial(Dminus);
	    }

	    double TauYplus = 230.2;
	    double TauYminus = 32.7;
	    double TauXplus = 66.6;
	    double Cplus = 0.0618;
	    double Cminus = 0.01;
	    double Dplus = 0.1548;
	    double Dminus = 0.1771;
	};

	struct TPrePostStdpState: public IProtoSerial<NDnnProto::TPrePostStdpState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kPrePostStdpStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
			serial(Yplus);
			serial(Yminus);
			serial(X);
		}

		double Yplus = 0.0;
		double Yminus = 0.0;
		TActVector<double> X;
	};

	template <typename TNeuron, typename TWeightNormalizationType>
	class TPrePostStdp: public TLearningRule<TPrePostStdpConst, TPrePostStdpState, TNeuron, TWeightNormalizationType> {
	public:
		using TPar = TLearningRule<TPrePostStdpConst, TPrePostStdpState, TNeuron, TWeightNormalizationType>;

		void Reset() {
			TPar::s.Yplus = 0.0;
			TPar::s.Yminus = 0.0;
			TPar::s.X.resize(TPar::GetSynapses().size());
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
	        TPar::s.X[sp.SynapseId] += 1.0;
    	}

    	void CalculateDynamics(const TTime& t) {
    		auto& syns = TPar::GetMutSynapses();
    		const auto& norm = TPar::Norm();
    		const auto& neuron = TPar::Neuron();
    		
    		double YminusPre = TPar::s.Yminus;
    		if (neuron.Fired()) {
    			TPar::s.Yplus += 1.0;
    			TPar::s.Yminus += 1.0;
    		}
	        
	        auto synIdIt = TPar::s.X.abegin();
		    while (synIdIt != TPar::s.X.aend()) {
		    	if(std::fabs(TPar::s.X[synIdIt]) < 0.0001) {
                	TPar::s.X.SetInactive(synIdIt);
                } else {
	                const ui32& synapseId = *synIdIt;
	                auto& syn = syns.Get(synapseId);
	                double& P = syn.MutWeight();
	                double& q = syn.MutPostSynapticWeight();

	                double dq = TPar::c.Cplus * TPar::s.X[synIdIt] * YminusPre * neuron.Fired() * norm.Ltp(q) - TPar::c.Cminus * neuron.Fired() * norm.Ltd(q);
					
					q += norm.Derivative(std::abs(q-1.0), dq);

	                double xPre = syn.Fired() ? TPar::s.X[synIdIt] - 1.0 : TPar::s.X[synIdIt];

	                double dP = (
	                    TPar::c.Dplus  * xPre * TPar::s.Yplus * syn.Fired() * norm.Ltp(P) -
	                    TPar::c.Dminus * TPar::s.Yminus * TPar::s.Yplus * syn.Fired() * norm.Ltd(P)
	                );
	                
	                P += norm.Derivative(P, dP);
	                
	                TPar::s.X[synIdIt] += - t.Dt * TPar::s.X[synIdIt]/TPar::c.TauXplus;

	                ++synIdIt;
                }
            }
            
            TPar::s.Yplus += - t.Dt * TPar::s.Yplus/TPar::c.TauYplus;
            TPar::s.Yminus += - t.Dt * TPar::s.Yminus/TPar::c.TauYminus;
    	}
	};

} // namespace NDnn
