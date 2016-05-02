#pragma once

#include <ground/serial/proto_serial.h>
#include <dnn/protos/config.pb.h>
#include <ground/ptr.h>

namespace NDnn {


    template <typename TConstants, typename TNeuronImpl>
    class TReinforcement: public IProtoSerial<NDnnProto::TLayer> {
    public:
        using TNeuronType = TNeuronImpl;

        void SerialProcess(TProtoSerial& serial) override final {
            serial(c, TConstants::ProtoFieldNumber);
        }

        void SetNeuronImpl(TNeuronImpl& neuron) {
            NeuronImpl.Set(neuron);
        }

        const typename TNeuronImpl::TNeuronType& Neuron() const {
            return NeuronImpl->GetNeuron();
        }

        const TNeuronSpaceInfo& SpaceInfo() const {
            return NeuronImpl->GetSpaceInfo();
        }

    protected:
        TConstants c;

        TPtr<TNeuronImpl> NeuronImpl;
    };


} // namespace NDnn
