#include "intrinsic_plasticity.h"

#include <dnn/protos/max_entropy.pb.h>
#include <dnn/protos/config.pb.h>

#include <dnn/activation/sigmoid.h>

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

	template <typename TNeuron>
	class TMaxEntropyIP: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPState, TNeuron> {
	public:
		using TPar = TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPState, TNeuron>;
		
		static_assert(std::is_same<typename TNeuron::TConfig::TNeuronActivationFunction, TSigmoid>::value, 
			"Max entropy intrinsic plasticity works only with sigmoid function activation");
		
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



} // namespace NDnn