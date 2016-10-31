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
            serial(TauSpike);
            serial(Aminus);
	    }

        double TauLearn = 20.0;
        NGroundProto::TDistr ErrorWeightsDistr;
        double Bias = 0.0;
        double TauSpike = 25.0;
        double Aminus = 1.0;
	};

	struct TResumeHiddenRuleState: public IProtoSerial<NDnnProto::TResumeHiddenRuleState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kResumeHiddenRuleStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
            serial(ErrorWeights);
            serial(ErrorTrace);
            serial(Y);
		}

        TVector<double> ErrorWeights;
        double ErrorTrace = 0.0;
        double Y = 0.0;
	};

    #define sign(a) ( ( (a) < 0 )  ?  -1.0  : 1.0 )

	template <typename TNeuron>
	class TResumeHiddenRule: public TLearningRule<TResumeHiddenRuleConst, TResumeHiddenRuleState, TNeuron> {
	public:
		using TPar = TLearningRule<TResumeHiddenRuleConst, TResumeHiddenRuleState, TNeuron>;

		void Reset() {
            if (TPar::s.ErrorWeights.empty()) {
                TVector<double> signs = TGlobalCtx::Inst().GetConnectionSign(TPar::SpaceInfo().GlobalId);
                // for (ui32 errorId=0; errorId < signs.size(); ++errorId) {
                //     TPar::s.ErrorWeights.push_back(std::fabs(TPar::Neuron().GetRand()->DrawValue(TPar::c.ErrorWeightsDistr)) * signs[errorId]);
                // }
                if (signs.size() > 0) {
                    Sign = signs[0];
                    for (const auto& sign: signs) {
                        ENSURE(std::fabs(sign - Sign) < std::numeric_limits<double>::epsilon(), "Got mixed connection for neuron " << TPar::SpaceInfo().GlobalId);
                    }
                    L_INFO << TPar::SpaceInfo().GlobalId << " " << Sign;
                }
            }
            TPar::s.Y = 0.0;
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
    	}

    	void CalculateDynamics(const TTime& t) {
    		auto& syns = TPar::GetMutSynapses();
    		const auto& norm = TPar::Norm();
    		const auto& neuron = TPar::Neuron();

            double error = 0.0;
      //       const auto& errors = TGlobalCtx().Inst().GetCausedErrors(TPar::SpaceInfo().GlobalId);
    		// for (ui32 errorId=0; errorId < TPar::s.ErrorWeights.size(); ++errorId) {
      //           error += TPar::s.ErrorWeights[errorId] * errors.at(errorId);
      //       }

            if (neuron.Fired()) {
                TPar::s.Y += 1.0;
            }

            error = TGlobalCtx::Inst().GetMeanError() * Sign;
            
            TGlobalCtx::Inst().SetError(TPar::SpaceInfo().GlobalId, error);

            CurrentError = error;

            TPar::s.ErrorTrace += t.Dt * ( - TPar::s.ErrorTrace/TPar::c.TauLearn + error );
            // L_INFO << TPar::SpaceInfo().GlobalId;
            auto synIdIt = syns.abegin();
            while (synIdIt != syns.aend()) {
                auto& syn = syns[synIdIt];

                double& w = syn.MutWeight();

                w += norm.Derivative(w, syn.LearningRate() * (TPar::c.Bias + syn.Potential() * TPar::s.ErrorTrace - TPar::c.Aminus * TPar::s.Y * syn.Fired() * norm.Ltd(w)));

                ++synIdIt;
            }

            TPar::s.Y += - t.Dt * TPar::s.Y/TPar::c.TauSpike;
    	}

        const double& GetCurrentError() const {
            return CurrentError;
        }

    private:
        double CurrentError = 0.0;
        double Sign = 1.0;

	};

} // namespace NDnn

