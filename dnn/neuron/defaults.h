#pragma once

#include <ground/serial/proto_serial.h>

namespace NDnn {


	template <typename TConstants>
	class TReceptiveField;

	template <typename TConstants, typename TState, typename TNeuronImpl>
	class TLearningRule;

	template <typename TConstants, typename TState, typename TNeuronImpl>
	class TWeightNormalization;

	template <typename TConstants, typename TState, typename TNeuronImpl>
	class TIntrinsicPlasticity;

	template <typename TConstants, typename TNeuronImpl>
	class TReinforcement;

	template <typename TConstants, typename TNeuronImpl>
	class TCostFunction;

	using TNoInput = TReceptiveField<TEmpty>;

	template <typename TNeuron>
	using TNoLearning = TLearningRule<TEmpty, TEmpty, TNeuron>;

	template <typename T>
	using TNoWeightNormalization = TWeightNormalization<TEmpty, TEmpty, T>;

	template <typename T>
	using TNoIntrinsicPlasticity = TIntrinsicPlasticity<TEmpty, TEmpty, T>;

	template <typename T>
	using TNoReinforcement = TReinforcement<TEmpty, T>;

	template <typename T>
	using TNoCostFunction = TCostFunction<TEmpty, T>;

	template <>
	class TReceptiveField<TEmpty>: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) {
		}

		void Init(const TNeuronSpaceInfo&, TRandEngine&) {
        }

        double CalculateResponse(double I) const {
            return 0.0;
        }
	};

	template <typename TNeuronImpl>
	class TLearningRule<TEmpty, TEmpty, TNeuronImpl>: public IProtoSerial<NDnnProto::TLayer> {
	public:
		using TWeightNormalizationType = typename TNeuronImpl::TConfig::template TWeightNormalization<TNeuronImpl>;

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

		void Reset() {
		}

	};

	template <typename T>
	class TIntrinsicPlasticity<TEmpty, TEmpty, T>: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) {
		}

		void Reset() {
		}

		void CalculateDynamics(const TTime& t) {
		}

		void SetNeuronImpl(T& neuron) {
		}
	};

	template <typename T>
	class TReinforcement<TEmpty, T>: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) {
		}

		void ModulateReward(const TTime&) {
		}

		void SetNeuronImpl(T& neuron) {
		}
	};

	template <typename T>
	class TCostFunction<TEmpty, T>: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) {
		}

		void CalculateError() {
		}

		void SetNeuronImpl(T& neuron) {
		}
	};

} // namespace NDnn