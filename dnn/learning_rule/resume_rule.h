#pragma once

#include "learning_rule.h"

#include <dnn/protos/config.pb.h>
#include <ground/act_vector.h>

namespace NDnn {

	struct TResumeRuleConst: public IProtoSerial<NDnnProto::TResumeRuleConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kResumeRuleFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
            serial(TauLearn);
            serial(Bias);
	    }

        double TauLearn = 20.0;
        double Bias = 0.0;

	};

	struct TResumeRuleState: public IProtoSerial<NDnnProto::TResumeRuleState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kResumeRuleStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
            serial(TargetTrace);
            serial(ActualTrace);
		}

        double TargetTrace = 0.0;
        double ActualTrace = 0.0;
	};

    #define sign(a) ( ( (a) < 0 )  ?  -1.0  : 1.0 )

	template <typename TNeuron>
	class TResumeRule: public TLearningRule<TResumeRuleConst, TResumeRuleState, TNeuron> {
	public:
		using TPar = TLearningRule<TResumeRuleConst, TResumeRuleState, TNeuron>;

		void Reset() {
            CurrentId = 0;
            TPar::s.TargetTrace = 0.0;
            TPar::s.ActualTrace = 0.0;
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
    	}

    	void CalculateDynamics(const TTime& t) {
    		auto& syns = TPar::GetMutSynapses();
    		const auto& norm = TPar::Norm();
    		const auto& neuron = TPar::Neuron();
    		
            TPar::s.TargetTrace += t.Dt * ( - TPar::s.TargetTrace/TPar::c.TauLearn + GetTarget(t) );
            TPar::s.ActualTrace += t.Dt * ( - TPar::s.ActualTrace/TPar::c.TauLearn + neuron.Fired() );

            const double error = TPar::s.TargetTrace - TPar::s.ActualTrace;

            TGlobalCtx::Inst().SetError(TPar::SpaceInfo().GlobalId, error);

            CurrentError = error;

            auto synIdIt = syns.abegin();
            while (synIdIt != syns.aend()) {
                auto& syn = syns[synIdIt];

                double& w = syn.MutWeight();
                
                w += norm.Derivative(w, syn.LearningRate() * (TPar::c.Bias + syn.Potential()) * error);

                ++synIdIt;
            }            
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

        const double& GetCurrentError() const {
            return CurrentError;
        }

    private:
    	bool TargetSet = false;
    	TVector<double> Target;
    	mutable ui32 CurrentId = 0;
        double CurrentError = 0.0;
	};

} // namespace NDnn
