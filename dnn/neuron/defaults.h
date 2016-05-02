#pragma once	

#include <ground/serial/proto_serial.h>

namespace NDnn {



	using TNoInput = TEmpty;

	template <typename T>
	using TNoReinforcement = TEmpty;


	template <typename TConstants, typename TState, typename TNeuronImpl, typename TWeightNormalizationType>
	class TLearningRule;


	template <typename X, typename Y, typename T>
	class TWeightNormalization;

	template <typename T>
	class TWeightNormalization<TEmpty, TEmpty, T>: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) {
		}

		double Ltp(const double& w) const {
			return 1.0;
		}

		double Ltd(const double& w) const {
			return 1.0;
		}

		double Derivative(double w, double dw) const {
			return dw;
		}

		void CalculateDynamics(const TTime& t) {
		}
		
		void SetNeuronImpl(T& neuron) {
		}

	};

	template <typename T>
	using TNoWeightNormalization = TWeightNormalization<TEmpty, TEmpty, T>;


	template <typename TNeuronImpl, typename TWeightNormalizationType>
	class TLearningRule<TEmpty, TEmpty, TNeuronImpl, TWeightNormalizationType>: public IProtoSerial<NDnnProto::TLayer> {
	public:
		static_assert(std::is_same<TWeightNormalizationType, TNoWeightNormalization<TNeuronImpl>>::value,
			"Trying to use weight normalization with empty learning rule");
		
		void SerialProcess(TProtoSerial& serial) override final {
		}

		void SetNeuronImpl(TNeuronImpl& neuron) {
		}

		const typename TNeuronImpl::TNeuronType& Neuron() const {
			throw TErrException() << "Not implemented";
		}

		const auto& GetSynapses() const {
			throw TErrException() << "Not implemented";
		}

		auto& GetMutSynapses() {
			throw TErrException() << "Not implemented";
		}

		const TEmpty& State() const {
		}
		
		const TNeuronSpaceInfo& SpaceInfo() const {
			throw TErrException() << "Not implemented";
		}

		const TWeightNormalizationType& Norm() const {
			return WeightNormalization;
		}

		TWeightNormalizationType& MutNorm() {
			return WeightNormalization;
		}

		void Reset() {
		}

		void PropagateSynapseSpike(const TSynSpike& sp) {
		}

		void CalculateDynamics(const TTime& t) {
		}
	private:
		TWeightNormalizationType WeightNormalization;
	};

	template <typename T3, typename T4>
	using TNoLearning = TLearningRule<TEmpty, TEmpty, T3, T4>;




} // namespace NDnn