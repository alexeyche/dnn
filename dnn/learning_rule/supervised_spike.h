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
            
            double error = fired - p;

    		auto synIdIt = TPar::s.FirstMoment.abegin();
		    while (synIdIt != TPar::s.FirstMoment.aend()) {
				const ui32& synapseId = *synIdIt;
                
                auto& syn = syns.Get(synapseId);
                double& w = syn.MutWeight();
                double dw = (p_deriv/(p/M)) * syn.Potential() * error;
                
                // TGlobalCtx::SetGrad(t, syn.IdPre(), dw);

                // L_INFO << "(" << p_deriv << "/" << p/M << ") * " << syn.Potential() << " * (" << fired << " - " << p << ")" << " -> " << dw;    
                
                TPar::s.FirstMoment[synIdIt] += t.Dt * ( - TPar::s.FirstMoment[synIdIt] + dw)/TPar::c.TauFirstMoment;
                TPar::s.SecondMoment.Get(synapseId) += t.Dt * ( - TPar::s.SecondMoment.Get(synapseId) + dw*dw)/TPar::c.TauSecondMoment;

                // TPar::s.FirstMoment[synIdIt] = dw;

                // w += norm.Derivative(w, 
                //     syn.LearningRate() * TPar::s.FirstMoment.Get(synapseId)/std::sqrt(TPar::s.SecondMoment.Get(synapseId) + 1e-08));
                w += norm.Derivative(w, syn.LearningRate() * TPar::s.FirstMoment[synIdIt]);
            	
                
            	if ((std::fabs(TPar::s.FirstMoment[synIdIt]) < 1e-06)&&(std::fabs(TPar::s.SecondMoment.Get(synapseId)) < 1e-06)) {
                	TPar::s.FirstMoment.SetInactive(synIdIt);
                } else {
	                ++synIdIt;
                }
            }
            
            // TPar::MutNeuron().MutSpikeProbability() = act.SpikeProbability(error) * neuron.ProbabilityModulation();
            // if(t.Dt * neuron.SpikeProbability() > neuron.GetRand()->GetUnif()) {
            //     TPar::MutNeuron().MutFired() = true;
            // } else {
            //     TPar::MutNeuron().MutFired() = false;
            // }


            TGlobalCtx::Inst().SetError(TPar::SpaceInfo().GlobalId, error*error);
    	}

    	double GetTarget(const TTime& t) const {
    		if (!TargetSet) {
    			return static_cast<double>(TPar::Neuron().Fired());
    		}
            double val = 0.0;
            // if (CurrentId < Target.size()) L_INFO << CurrentId << " " << Target.size() << " " << t.T << " >= " << Target[CurrentId] << ", " << t.T+t.Dt << " < " << Target[CurrentId]; 
    		while ((CurrentId < Target.size()) && (std::numeric_limits<double>::epsilon() + Target[CurrentId] >= t.T) && (Target[CurrentId] < (t.T+t.Dt))) {
                ++CurrentId;
    			val = 1.0;
    		}
            // L_INFO << t.T << " " << val;
    		return val;
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
