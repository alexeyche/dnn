#pragma once

#include <ground/serial/proto_serial.h>
#include <dnn/protos/config.pb.h>
#include <ground/ptr.h>
#include <dnn/neuron/spike_neuron_impl.h>

namespace NDnn {
	using namespace NGround;

	template <typename TConstants, typename TState, typename TNeuronImpl>
	class TIntrinsicPlasticity: public IProtoSerial<NDnnProto::TLayer> {
	public:
		using TNeuronType = TNeuronImpl;

		void SerialProcess(TProtoSerial& serial) override final {
			serial(c, TConstants::ProtoFieldNumber);
			serial(s, TState::ProtoFieldNumber);
		}

		void SetNeuronImpl(TNeuronImpl& neuron) {
			NeuronImpl.Set(neuron);
		}

		const typename TNeuronImpl::TNeuronType& Neuron() const {
			return NeuronImpl->GetNeuron();
		}

		typename TNeuronImpl::TConfig::TNeuronActivationFunction& MutActivationFunction() {
			return NeuronImpl->GetMutActivationFunction();
		}

		typename TNeuronImpl::TConfig::template TWeightNormalization<TNeuronImpl>& MutWeightNormalization() {
			return NeuronImpl->GetMutWeightNormalization();
		}

		const TState& State() const {
			return s;
		}
		
	protected:
		TConstants c;
		TState s;

	private:
		TPtr<TNeuronImpl> NeuronImpl;
	};


} // namespace NDnn
