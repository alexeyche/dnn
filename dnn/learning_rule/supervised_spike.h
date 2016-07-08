#pragma once

#include "learning_rule.h"

#include <dnn/protos/supervised_spike.pb.h>
#include <dnn/protos/config.pb.h>
#include <ground/act_vector.h>

namespace NDnn {

	struct TSupervisedSpikeConst: public IProtoSerial<NDnnProto::TSupervisedSpikeConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSupervisedSpikeFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	        serial(TauFirstMoment);
	        serial(TauSecondMoment);
	    }

	    double TauFirstMoment = 10.0;
	    double TauSecondMoment = 100.0;
	};

	struct TSupervisedSpikeState: public IProtoSerial<NDnnProto::TSupervisedSpikeState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kSupervisedSpikeStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
			serial(FirstMoment);
			serial(SecondMoment);
		}

		TActVector<double> FirstMoment;
    	TActVector<double> SecondMoment;
	};

	template <typename TNeuron>
	class TSupervisedSpike: public TLearningRule<TSupervisedSpikeConst, TSupervisedSpikeState, TNeuron> {
	public:
		using TPar = TLearningRule<TSupervisedSpikeConst, TSupervisedSpikeState, TNeuron>;

		void Reset() {
			TPar::s.FirstMoment.resize(TPar::GetSynapses().size(), 0.0);
			TPar::s.SecondMoment.resize(TPar::GetSynapses().size(), 0.0);
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
			TPar::s.FirstMoment.SetActive(sp.SynapseId);
    	}

    	void CalculateDynamics(const TTime& t) {
    		auto& syns = TPar::GetMutSynapses();
    		const auto& norm = TPar::Norm();
    		const auto& neuron = TPar::Neuron();
    		const auto& act = TPar::ActivationFunction();
    		const double& p = neuron.SpikeProbability();
			const double p_deriv = act.SpikeProbabilityDerivative(neuron.Membrane());
			const double& M = neuron.ProbabilityModulation();
			const double fired = GetTarget(t);

    		auto synIdIt = TPar::s.FirstMoment.abegin();
		    while (synIdIt != TPar::s.FirstMoment.aend()) {
				const ui32& synapseId = *synIdIt;
                
                auto& syn = syns.Get(synapseId);
                double& w = syn.MutWeight();
                // L_INFO << "(" << p_deriv << "/" << p/M << ") * " << syn.Potential() << " * (" << fired << " - " << p << ")";
                double dw = (p_deriv/(p/M)) * syn.Potential() * (fired - p);

                TPar::s.FirstMoment[synIdIt] += t.Dt * ( - TPar::s.FirstMoment[synIdIt] + dw)/TPar::c.TauFirstMoment;
                // TPar::s.FirstMoment[synIdIt] = dw;
                w += norm.Derivative(w, syn.LearningRate() * TPar::s.FirstMoment[synIdIt]);
            	
                
            	if (std::fabs(TPar::s.FirstMoment[synIdIt]) < 1e-06) {
                	TPar::s.FirstMoment.SetInactive(synIdIt);
                } else {
	                ++synIdIt;
                }
            }
    	}

    	double GetTarget(const TTime& t) const {
    		if (!TargetSet) {
    			return static_cast<double>(TPar::Neuron().Fired());
    		}
            // if (CurrentId < Target.size()) L_INFO << CurrentId << " " << Target.size() << " " << t.T << " >= " << Target[CurrentId] << ", " << t.T+t.Dt << " < " << Target[CurrentId]; 
    		if ((CurrentId < Target.size()) && (Target[CurrentId] >= t.T) && (Target[CurrentId] < (t.T+t.Dt))) {
                ++CurrentId;
    			return 1.0;
    		}
    		return 0.0;
    	}

    	void SetTarget(const TVector<double>& targetSeq) {
    		Target = targetSeq;
    		TargetSet = true;
    		CurrentId = 0;
    	}

    private:
    	bool TargetSet = false;
    	TVector<double> Target;
    	mutable ui32 CurrentId = 0;
	};

} // namespace NDnn
