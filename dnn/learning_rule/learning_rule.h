#pragma once

#include <ground/serial/proto_serial.h>
#include <dnn/protos/config.pb.h>
#include <ground/ptr.h>
#include <dnn/neuron/spike_neuron_impl.h>

namespace NDnn {
	using namespace NGround;

	template <typename TConstants, typename TState, typename TNeuronImpl>
	class TLearningRule: public IProtoSerial<NDnnProto::TLayer> {
	public:
		using TNeuronType = TNeuronImpl;
		using TWeightNormalizationType = typename TNeuronImpl::TConfig::template TWeightNormalization<TNeuronImpl>;

		void SerialProcess(TProtoSerial& serial) override final {
			serial(c, TConstants::ProtoFieldNumber);
			serial(s, TState::ProtoFieldNumber);
		}

		void SetNeuronImpl(TNeuronImpl& neuron) {
			NeuronImpl.Set(neuron);
			WeightNormalization.SetNeuronImpl(neuron);
		}

		const typename TNeuronImpl::TNeuronType& Neuron() const {
			return NeuronImpl->GetNeuron();
		}

		typename TNeuronImpl::TNeuronType& MutNeuron() {
			return NeuronImpl->GetNeuron();
		}

		const auto& GetSynapses() const {
			return NeuronImpl->GetSynapses();
		}

		auto& GetMutSynapses() {
			return NeuronImpl->GetMutSynapses();
		}

		const TState& State() const {
			return s;
		}
		
		const TNeuronSpaceInfo& SpaceInfo() const {
			return NeuronImpl->GetSpaceInfo();
		}

		const TWeightNormalizationType& Norm() const {
			return WeightNormalization;
		}

		TWeightNormalizationType& MutNorm() {
			return WeightNormalization;
		}

		const typename TNeuronImpl::TConfig::TNeuronActivationFunction& ActivationFunction() {
			return NeuronImpl->GetActivationFunction();
		}

	protected:
		TConstants c;
		TState s;

		TWeightNormalizationType WeightNormalization;
	private:
		TPtr<TNeuronImpl> NeuronImpl;
	};


} // namespace NDnn
