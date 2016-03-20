#pragma once

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/config.pb.h>

namespace NDnn {


	template <typename TConstants, typename TState, typename TNeuronImpl>
	class TLearningRule: public IProtoSerial<NDnnProto::TLayer> {
	public:
		using TNeuronType = TNeuronImpl;

		void SerialProcess(TProtoSerial& serial) override final {
			serial(c, TConstants::ProtoFieldNumber);
			serial(s, TState::ProtoFieldNumber);
		}

		void SetNeuronImpl(TNeuronImpl& neuron) {
			Neuron.Set(neuron);
		}

		typename TNeuronImpl::TNeuronType& GetNeuron() {
			return Neuron->GetNeuron();
		}

		const auto& GetSynapses() const {
			return Neuron->GetSynapses();
		}

		auto& GetMutSynapses() {
			return Neuron->GetMutSynapses();
		}

		const TState& State() const {
			return s;
		}

	protected:
		TConstants c;
		TState s;

	private:
		TPtr<TNeuronImpl> Neuron;
	};


} // namespace NDnn
