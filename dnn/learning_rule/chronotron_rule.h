#pragma once

#include "learning_rule.h"

#include <dnn/protos/chronotron_rule.pb.h>
#include <dnn/protos/config.pb.h>
#include <ground/act_vector.h>

namespace NDnn {

	struct TChronotronRuleConst: public IProtoSerial<NDnnProto::TChronotronRuleConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kChronotronRuleFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	    }

	};

	struct TChronotronRuleState: public IProtoSerial<NDnnProto::TChronotronRuleState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kChronotronRuleStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
		}
	};

    #define sign(a) ( ( (a) < 0 )  ?  -1.0  : 1.0 )

	template <typename TNeuron>
	class TChronotronRule: public TLearningRule<TChronotronRuleConst, TChronotronRuleState, TNeuron> {
	public:
		using TPar = TLearningRule<TChronotronRuleConst, TChronotronRuleState, TNeuron>;

		void Reset() {
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
    	}

    	void CalculateDynamics(const TTime& t) {
    		auto& syns = TPar::GetMutSynapses();
    		const auto& norm = TPar::Norm();
    		const auto& neuron = TPar::Neuron();
    		
            const double t_fired = GetTarget(t);
            const double fired = neuron.Fired();

            double error = t_fired - neuron.Membrane();
            
            auto synIdIt = syns.abegin();
            while (synIdIt != syns.aend()) {
                auto& syn = syns[synIdIt];

                double& w = syn.MutWeight();
                const double I = syn.WeightedPotential();

                w += syn.LearningRate() * sign(w) * (t_fired * I - fired * I);

                ++synIdIt;
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
