#pragma once

#include <dnn/util/serial/proto_serial.h>
#include <dnn/protos/config.pb.h>
#include <dnn/util/ptr.h>

namespace NDnn {


	template <typename TConstants, typename TState, typename TNeuronImpl>
	class TWeightNormalization: public IProtoSerial<NDnnProto::TLayer> {
	public:
		void SerialProcess(TProtoSerial& serial) override final {
			serial(c, TConstants::ProtoFieldNumber);
			serial(s, TState::ProtoFieldNumber);
		}

		const TState& State() const {
			return s;
		}

		const TConstants& Const() const {
			return c;
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
		
		void SetNeuronImpl(TNeuronImpl& neuron) {
			NeuronImpl.Set(neuron);
		}

		auto& GetMutSynapses() {
			return NeuronImpl->GetMutSynapses();
		}

		const typename TNeuronImpl::TNeuronType& Neuron() const {
			return NeuronImpl->GetNeuron();
		}
		
	protected:
		TConstants c;
		TState s;
	private:
		TPtr<TNeuronImpl> NeuronImpl;
	};



} // namespace NDnn
