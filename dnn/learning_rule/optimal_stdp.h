#pragma once

#include "learning_rule.h"

#include <dnn/protos/optimal_stdp.pb.h>
#include <dnn/protos/config.pb.h>
#include <ground/act_vector.h>
#include <dnn/sim/global_ctx.h>

namespace NDnn {

	struct TOptimalStdpConst: public IProtoSerial<NDnnProto::TOptimalStdpConst> {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kOptimalStdpFieldNumber;

	    void SerialProcess(TProtoSerial& serial) {
	        serial(TauC);
		    serial(TauMean);
		    serial(TargetRate);
		    serial(TargetRate);
		    serial(WeightDecay);
		    serial(LearningRate);
		    serial(TauHebb);

		    __TargetRate = TargetRate/1000.0;
	    }

	    double TauC = 100.0;
	    double TauMean = 10000.0;
	    double TargetRate = 10.0;
	    double __TargetRate;
	    double TargetRateFactor = 1.0;
	    double WeightDecay = 0.0026;
	    double LearningRate = 0.01;
	    double TauHebb = 0.0;
	};

	struct TOptimalStdpState: public IProtoSerial<NDnnProto::TOptimalStdpState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kOptimalStdpStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
			serial(C);
			serial(B);
			serial(Pmean);
		}

    	TActVector<double> C;
    	double B = 0.0;
    	double Pmean = 0.0;
	};

	template <typename TNeuron>
	class TOptimalStdp: public TLearningRule<TOptimalStdpConst, TOptimalStdpState, TNeuron> {
	public:
		using TPar = TLearningRule<TOptimalStdpConst, TOptimalStdpState, TNeuron>;

		void Reset() {
			TPar::s.B = 0.0;
			TPar::s.C.resize(TPar::GetSynapses().size(), 0.0);
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
			TPar::s.C.SetActive(sp.SynapseId);
    	}

    	void CalculateDynamics(const TTime& t) {
    		auto& syns = TPar::GetMutSynapses();
    		const auto& norm = TPar::Norm();
    		const auto& neuron = TPar::Neuron();
    		const auto& act = TPar::ActivationFunction();

    		const double& p = neuron.SpikeProbability();
    		const double& M = neuron.ProbabilityModulation();
			const double p_deriv = act.SpikeProbabilityDerivative(neuron.Membrane());
			const double fired = static_cast<double>(neuron.Fired());

			TPar::s.Pmean += (-TPar::s.Pmean + fired)/TPar::c.TauMean;

		    if( (TGlobalCtx::Inst().GetPastTime() + t.T) < TPar::c.TauMean ) {
		        return;
		    }

			if (std::fabs(TPar::s.Pmean) > 1e-04) {
		        TPar::s.B = (( fired * std::log(p/TPar::s.Pmean) - (p - TPar::s.Pmean)) -  \
		            TPar::c.TargetRateFactor * (fired* std::log(TPar::s.Pmean/TPar::c.__TargetRate) - (TPar::s.Pmean - TPar::c.__TargetRate)) );
		    } else {
		        TPar::s.B = 0.0;
		    }

			auto synIdIt = TPar::s.C.abegin();
		    while (synIdIt != TPar::s.C.aend()) {
		    	const ui32& synapseId = *synIdIt;
                auto& syn = syns.Get(synapseId);
                double& w = syn.MutWeight();

                TPar::s.C[synIdIt] += (p_deriv/(p/M)) * syn.Potential() * (norm.Ltp(w) * fired - norm.Ltd(w) * p/(1.0+TPar::c.TauHebb*p));

                double dw = TPar::c.LearningRate * (TPar::s.C[synIdIt] * TPar::s.B - TPar::c.WeightDecay * syn.Fired() * syn.Weight());

                w += norm.Derivative(w, dw);

                if (std::isnan(w)) {
		            std::cerr << fired << " * log(" << p << "/" << TPar::s.Pmean << ")" << " - (" << p << " - " << TPar::s.Pmean << ")\n";
		            throw TErrException() << "Found nan weight. B: " << TPar::s.B << ", C: " << TPar::s.C[synIdIt] << "\n";
		        }

		        TPar::s.C[synIdIt] += - t.Dt * TPar::s.C[synIdIt]/TPar::c.TauC;

		    	if (std::fabs(TPar::s.C[synIdIt]) < 0.0001) {
                	TPar::s.C.SetInactive(synIdIt);
                } else {
                	++synIdIt;
                }
            }
    	}
	};

} // namespace NDnn
