#pragma once

#include "learning_rule.h"

#include <dnn/protos/config.pb.h>
#include <ground/act_vector.h>
#include <ground/protos/distr.pb.h>

namespace NDnn {

	struct TResumeHiddenRuleConst: public IProtoSerial<NDnnProto::TResumeHiddenRuleConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kResumeHiddenRuleFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
            serial(TauLearn);
            serial(ErrorWeightsDistr);
            serial(Bias);
	    }

        double TauLearn = 20.0;
        NGroundProto::TDistr ErrorWeightsDistr;
        double Bias = 0.0;
	};

	struct TResumeHiddenRuleState: public IProtoSerial<NDnnProto::TResumeHiddenRuleState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kResumeHiddenRuleStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
            serial(ErrorWeights);
            serial(ErrorTrace);
		}

        TVector<double> ErrorWeights;
        double ErrorTrace = 0.0;
	};

    #define sign(a) ( ( (a) < 0 )  ?  -1.0  : 1.0 )

	template <typename TNeuron>
	class TResumeHiddenRule: public TLearningRule<TResumeHiddenRuleConst, TResumeHiddenRuleState, TNeuron> {
	public:
		using TPar = TLearningRule<TResumeHiddenRuleConst, TResumeHiddenRuleState, TNeuron>;

		void Reset() {
            if (TPar::s.ErrorWeights.empty()) {
                TVector<double> signs = TGlobalCtx::Inst().GetConnectionSign(TPar::SpaceInfo().GlobalId);
                for (ui32 errorId=0; errorId < signs.size(); ++errorId) {
                    TPar::s.ErrorWeights.push_back(std::fabs(TPar::Neuron().GetRand()->DrawValue(TPar::c.ErrorWeightsDistr)) * signs[errorId]);
                }
            }
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
    	}

    	void CalculateDynamics(const TTime& t) {
    		auto& syns = TPar::GetMutSynapses();
    		const auto& norm = TPar::Norm();
    		const auto& neuron = TPar::Neuron();

            double error = 0.0;
            const auto& errors = TGlobalCtx().Inst().GetCausedErrors(TPar::SpaceInfo().GlobalId);
    		for (ui32 errorId=0; errorId < TPar::s.ErrorWeights.size(); ++errorId) {
                error += TPar::s.ErrorWeights[errorId] * errors.at(errorId);
            }

            TGlobalCtx::Inst().SetError(TPar::SpaceInfo().GlobalId, error);

            CurrentError = error;

            TPar::s.ErrorTrace += t.Dt * ( - TPar::s.ErrorTrace/TPar::c.TauLearn + error );

            auto synIdIt = syns.abegin();
            while (synIdIt != syns.aend()) {
                auto& syn = syns[synIdIt];

                double& w = syn.MutWeight();

                w += norm.Derivative(w, syn.LearningRate() * (TPar::c.Bias + syn.Potential()) * TPar::s.ErrorTrace);

                ++synIdIt;
            }
    	}

        const double& GetCurrentError() const {
            return CurrentError;
        }

    private:
        double CurrentError = 0.0;

	};

} // namespace NDnn

