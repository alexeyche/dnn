#include "intrinsic_plasticity.h"

#include <dnn/protos/max_entropy.pb.h>
#include <dnn/protos/config.pb.h>

#include <dnn/activation/determ.h>
#include <dnn/activation/sigmoid.h>
#include <dnn/activation/log_exp.h>

#include <dnn/weight_normalization/sum_norm.h>

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
	        serial(TauMoment);

	        __TargetRate = TargetRate/1000.0;
	    }

	    double TargetRate = 10.0;
	    double LearningRate = 0.01;
	    double TauMoment = 1000.0;
	    double __TargetRate;
	};

	struct TMaxEntropyIPDefaultState: public IProtoSerial<NDnnProto::TMaxEntropyIPDefaultState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kMaxEntropyIPDefaultStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
		}
	};

	template <typename T> 
	constexpr bool DependentFalse() {
	    return false;
	}


	template <typename TNeuron, typename TActivation, typename TWeightNormalizationType>
	class TMaxEntropyIPImpl: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPDefaultState, TNeuron> {
	public:
		static_assert(DependentFalse<TActivation>(), "This activation function is not supported by max entropy intrinsic plasticity");

		void Reset() {}
		void CalculateDynamics(const TTime&) {}
	};

	template <typename TNeuron, typename TWeightNormalizationType>
	class TMaxEntropyIPImpl<TNeuron, TSigmoid, TWeightNormalizationType>: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPDefaultState, TNeuron> {
	public:
		using TPar = TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPDefaultState, TNeuron>;
		
		void Reset() {
			TSigmoid& actF = static_cast<TSigmoid&>(TPar::MutActivationFunction());
			
			ExpThreshold.Set(&actF.MutConstants().Threshold);
			ExpSlope.Set(&actF.MutConstants().Slope);

			ENSURE(*ExpThreshold > 0.0, "Need Threshold in Sigmoid activation be >= 0.0");
			ENSURE(*ExpSlope > 0.0, "Need Slope in Sigmoid activation be >= 0.0");
			
			Threshold = std::log(*ExpThreshold);
			Slope = std::log(*ExpSlope);
		}

    	void CalculateDynamics(const TTime& t) {
    		const auto& x = TPar::Neuron().Membrane();
    		const double& y = TPar::Neuron().SpikeProbability();
    		const double twoY = 2.0 * y;
    		const auto& mu = TPar::c.__TargetRate;
    		const double mu_part = (1.0/mu) * (y * (1.0 - y));
    		
    		const double expT = std::exp(Threshold);
    		const double expS = std::exp(Slope);

    		const double dSlope = - TPar::c.LearningRate * ((x - expT)/expS) * ( twoY + mu_part);
    		const double dThreshold = TPar::c.LearningRate * (std::exp(Threshold - Slope)) * (twoY - 1.0 + mu_part);
    		    		
    		Slope += t.Dt * dSlope;
    		Threshold += t.Dt * dThreshold;

    		*ExpSlope = expS;
    		*ExpThreshold = expT;

    		L_DEBUG << expS << " " << expT;
    	}

    private:
    	TPtr<double> ExpThreshold;
    	TPtr<double> ExpSlope;

    	double Threshold;
    	double Slope;
	};

	#define LOG_EXP_RESET() { \
		TLogExp& actF = static_cast<TLogExp&>(TPar::MutActivationFunction()); \
		\
		ExpThreshold.Set(&actF.MutConstants().Threshold); \
		ExpSlope.Set(&actF.MutConstants().Slope); \
		\
		ENSURE(*ExpThreshold > 0.0, "Need Threshold in LogExp activation be >= 0.0"); \
		ENSURE(*ExpSlope > 0.0, "Need Slope in LogExp activation be >= 0.0"); \
		\
		Threshold = std::log(*ExpThreshold); \
		Slope = std::log(*ExpSlope); \
	}

	#define LOG_EXP_CALCULATIONS() { \
		const auto& x = TPar::Neuron().Membrane(); \
		const double& y = TPar::Neuron().SpikeProbability(); \
		const auto& mu = TPar::c.__TargetRate; \
		\
		const double expT = std::exp(Threshold); \
		const double expS = std::exp(Slope); \
		const double x1 = (x - expT)/expS; \
		const double x0 = - expT/expS; \
		const double exp_x1 = std::exp(x1); \
		const double exp_x0 = std::exp(x0); \
		const double x1val = exp_x1 / (exp_x1 + 1.0); \
		const double x0val = exp_x0 / (exp_x0 + 1.0); \
		\
		double dSlope = - TPar::c.LearningRate * (x1val * (-x1) + x1 + 1.0 + (1.0/mu) * (x1val * (-x1) - x0val * (-x0))); \
		\
		double dThreshold = - TPar::c.LearningRate * ( -x0 + x1val * x0 + (1.0/mu) * (x1val * x0 - x0val * x0)); \
		\
		Slope += t.Dt * dSlope; \
		Threshold += t.Dt * dThreshold; \
		\
		*ExpSlope = expS; \
		*ExpThreshold = expT; \
	}

	template <typename TNeuron, typename TWeightNormalizationType>
	class TMaxEntropyIPImpl<TNeuron, TLogExp, TWeightNormalizationType>: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPDefaultState, TNeuron> {
	public:
		using TPar = TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPDefaultState, TNeuron>;
		
		void Reset() {
			LOG_EXP_RESET();
		}

    	void CalculateDynamics(const TTime& t) {
    		LOG_EXP_CALCULATIONS();
    	}

    private:
    	TPtr<double> ExpThreshold;
    	TPtr<double> ExpSlope;

    	double Threshold;
    	double Slope;
	};

	struct TMaxEntropyIPMomentState: public IProtoSerial<NDnnProto::TMaxEntropyIPMomentState>  {
		static const auto ProtoFieldNumber = NDnnProto::TLayer::kMaxEntropyIPMomentStateFieldNumber;

		void SerialProcess(TProtoSerial& serial) {
			serial(M1);
			serial(M2);
		}

		double M1 = 0.0;
		double M2 = 0.0;
	};


	template <typename TNeuron>
	class TMaxEntropyIPImpl<TNeuron, TLogExp, TSumNorm<TNeuron>>: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPMomentState, TNeuron> {
	public:
		using TPar = TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPMomentState, TNeuron>;
		
		void Reset() {
			LOG_EXP_RESET();
			TSumNorm<TNeuron>& weightNorm = static_cast<TSumNorm<TNeuron>&>(TPar::MutWeightNormalization());
			
			ExpExcUnit.Set(&weightNorm.MutConstants().ExcUnit);
			ExpInhUnit.Set(&weightNorm.MutConstants().InhUnit);
			
			ExcUnit = std::log(*ExpExcUnit);
			InhUnit = std::log(*ExpInhUnit);
		}

    	void CalculateDynamics(const TTime& t) {
    		LOG_EXP_CALCULATIONS();
		    if( (TGlobalCtx::Inst().GetPastTime() + t.T) < 10.0*TPar::c.TauMoment ) {
		        return;
		    }

    		const auto& mu = TPar::c.__TargetRate;
		    TPar::s.M1 += t.Dt * (-TPar::s.M1 + TPar::Neuron().SpikeProbability())/TPar::c.TauMoment;
		    
		    double deriv = 1000*TPar::c.LearningRate * (TPar::s.M1 - mu);

		    ExcUnit += - deriv;
		    InhUnit += deriv;

		    *ExpExcUnit = std::exp(ExcUnit);
		    *ExpInhUnit = std::exp(InhUnit);
		    // L_INFO << "deriv: " << deriv << " M1: " << TPar::s.M1 << " M2: " << TPar::s.M2 << ", E: " << *ExcUnit << ", I: " << *InhUnit;
    	}

    private:
    	TPtr<double> ExpThreshold;
    	TPtr<double> ExpSlope;

    	double Threshold;
    	double Slope;

    	TPtr<double> ExpExcUnit;
    	TPtr<double> ExpInhUnit;

    	double ExcUnit;
    	double InhUnit;
	};


	// struct TMaxEntropyIPDetermState: public IProtoSerial<NDnnProto::TMaxEntropyIPDetermState>  {
	// 	static const auto ProtoFieldNumber = NDnnProto::TLayer::kMaxEntropyIPDetermStateFieldNumber;

	// 	void SerialProcess(TProtoSerial& serial) {
	// 		serial(M1);
	// 		serial(M2);
	// 	}

	// 	double M1 = 0.0;
	// 	double M2 = 0.0;
	// };


	// template <typename TNeuron>
	// class TMaxEntropyIPImpl<TNeuron, TDeterm>: public TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPDetermState, TNeuron> {
	// public:
	// 	using TPar = TIntrinsicPlasticity<TMaxEntropyIPConst, TMaxEntropyIPDetermState, TNeuron>;
		
	// 	void Reset() {
	// 		TDeterm& actF = static_cast<TDeterm&>(TPar::MutActivationFunction());
			
	// 		A.Set(&actF.MutConstants().A);
	// 		B.Set(&actF.MutConstants().B);
	// 	}

 //    	void CalculateDynamics(const TTime& t) {
 //    		const auto& mu = TPar::c.__TargetRate;
	// 		const double fired = static_cast<double>(TPar::Neuron().Fired());
			
	// 		TPar::s.M1 += (-TPar::s.M1 + fired)/TPar::c.TauMoment;
	// 		TPar::s.M2 += (-TPar::s.M2 + TPar::s.M1 * TPar::s.M1 * 2.0)/TPar::c.TauMoment;

 //    		double da = TPar::c.LearningRate * (TPar::s.M2 - 2.0 * mu * mu);
 //    		double db = TPar::c.LearningRate * (TPar::s.M1 - mu);
    		
 //    		*A += t.Dt * da;
 //    		*B += t.Dt * db;
 //    	}

 //    private:
 //    	TPtr<double> A;
 //    	TPtr<double> B;
	// };




	template <typename TNeuron>
	using TMaxEntropyIP = TMaxEntropyIPImpl<
		TNeuron, 
		typename TNeuron::TConfig::TNeuronActivationFunction, 
		typename TNeuron::TConfig::template TWeightNormalization<TNeuron>
	>;


} // namespace NDnn