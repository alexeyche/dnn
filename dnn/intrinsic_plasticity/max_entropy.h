#include "intrinsic_plasticity.h"

#include <dnn/protos/max_entropy.pb.h>
#include <dnn/protos/config.pb.h>

#include <dnn/activation/sigmoid.h>
#include <dnn/activation/log_exp.h>

#include <type_traits>

namespace NDnn {

	struct TMaxEntropyIPConst: public IProtoSerial<NDnnProto::TMaxEntropyIPConst> {
		TMaxEntropyIPConst()
			: __TargetRate(TargetRate/1000.0)
		{}

		static const auto ProtoFieldNumber = NDnnProto::TLayer::kMaxEntropyIPFieldNumber;
		
	    void SerialProcess(TProtoSerial& serial) {
	        serial(TargetRate);
	        serial(LearningRate);

	        __TargetRate = TargetRate/1000.0;
	    }

	    double TargetRate = 10.0;
	    double LearningRate = 0.01;
	    double __TargetRate;
	};

	struct TMaxEntropyIPState: public IProtoSerial<NDnnProto::TMaxEntropyIPState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kMaxEntropyIPStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
		}
	};

	template <typename T> 
	constexpr bool DependentFalse() {
	    return false;
	}


	template <typename TNeuron, typename TActivation>
	class TMaxEntropyIPImpl: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPState, TNeuron> {
	public:
		static_assert(DependentFalse<TActivation>(), "This activation function is not supported by max entropy intrinsic plasticity");

		void Reset() {}
		void CalculateDynamics(const TTime&) {}
	};

	template <typename TNeuron>
	class TMaxEntropyIPImpl<TNeuron, TSigmoid>: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPState, TNeuron> {
	public:
		using TPar = TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPState, TNeuron>;
		
		void Reset() {
			TSigmoid& actF = static_cast<TSigmoid&>(TPar::MutActivationFunction());
			
			A.Set(&actF.MutConstants().A);
			B.Set(&actF.MutConstants().B);
		}

    	void CalculateDynamics(const TTime& t) {
    		const auto& x = TPar::Neuron().Membrane();
    		double y = TPar::Neuron().SpikeProbability();
    		const auto& mu = TPar::c.__TargetRate;
    		double y2_on_mu = y*y/mu;
    		
    		double da = TPar::c.LearningRate * (1.0 / *A + x - ( 2.0 + 1.0/mu) * x * y + x*y2_on_mu);
    		double db = TPar::c.LearningRate * (1.0 - (2.0 + 1.0/mu) * y  + y2_on_mu);
    		
    		*A += t.Dt * da;
    		*B += t.Dt * db;
    	}

    private:
    	TPtr<double> A;
    	TPtr<double> B;
	};


	template <typename TNeuron>
	class TMaxEntropyIPImpl<TNeuron, TLogExp>: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPState, TNeuron> {
	public:
		using TPar = TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPState, TNeuron>;
		
		void Reset() {
			TLogExp& actF = static_cast<TLogExp&>(TPar::MutActivationFunction());
			
			R0.Set(&actF.MutConstants().R0);
			U0.Set(&actF.MutConstants().U0);
			Ua.Set(&actF.MutConstants().Ua);
		}

    	void CalculateDynamics(const TTime& t) {
    		const auto& x = TPar::Neuron().Membrane();
    		double y = TPar::Neuron().SpikeProbability();
    		const auto& mu = TPar::c.__TargetRate;
    		double y2_on_mu = y*y/mu;
    		
    		double some_val = (1.0  + *R0 / mu) * (1.0 - std::exp(- y / *R0)) - 1.0;

    		double dr0 = TPar::c.LearningRate * ((1.0 - y / mu) / *R0 );
    		double du0 = TPar::c.LearningRate * some_val / *U0;
    		double dua = TPar::c.LearningRate * (((x - *U0) / *Ua) * some_val - 1.0) / *Ua;
    		
    		// L_DEBUG << "x: " << x << "; " << *R0 << " + " << dr0 << "; " << *U0 << " + " << du0 << "; " << *Ua << " + " << dua;
    		
    		*R0 += t.Dt * dr0;
    		*U0 += t.Dt * du0;
    		*Ua += t.Dt * dua;
    	}

    private:
    	TPtr<double> R0;
    	TPtr<double> U0;
    	TPtr<double> Ua;
	};


	template <typename TNeuron>
	using TMaxEntropyIP = TMaxEntropyIPImpl<TNeuron, typename TNeuron::TConfig::TNeuronActivationFunction>;


} // namespace NDnn