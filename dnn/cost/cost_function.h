#pragma once

#include <ground/serial/proto_serial.h>
#include <dnn/protos/config.pb.h>
#include <ground/ptr.h>

namespace NDnn {


    template <typename TConstants, typename TNeuronImpl>
    class TCostFunction: public IProtoSerial<NDnnProto::TLayer> {
    public:
        using TNeuronType = TNeuronImpl;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(c, TConstants::ProtoFieldNumber);
        }

        void SetNeuronImpl(TNeuronImpl& neuron) {
            NeuronImpl.Set(neuron);
        }

        typename TNeuronImpl::TNeuronType& Neuron() {
            return NeuronImpl->GetNeuron();
        }

        const TNeuronSpaceInfo& SpaceInfo() const {
            return NeuronImpl->GetSpaceInfo();
        }

        auto& MutSynapses() {
            return NeuronImpl->GetMutSynapses();
        }

        const typename TNeuronImpl::TConfig::TNeuronActivationFunction& ActivationFunction() {
            return NeuronImpl->GetActivationFunction();
        }


    protected:
        TConstants c;

        TPtr<TNeuronImpl> NeuronImpl;
    };


} // namespace NDnn
