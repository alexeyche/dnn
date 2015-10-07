#pragma once

#include <dnn/io/serialize.h>
#include <dnn/base/exceptions.h>

namespace dnn {


struct ReinforcementInterface {
    stateDelegate modulateReward;
};


class ReinforcementBase : public SerializableBase {
public:
    typedef ReinforcementInterface interface;

    virtual void modulateReward() = 0;

    template <typename T>
    void provideInterface(ReinforcementInterface &i) {
        i.modulateReward = MakeDelegate(static_cast<T*>(this), &T::modulateReward);
    }
    static void __default() {}

    virtual void linkWithNeuron(SpikeNeuronBase &_n) = 0;

    static void provideDefaultInterface(ReinforcementInterface &i) {
        i.modulateReward = &ReinforcementBase::__default;
    }
};



template <typename Constants, typename Neuron = SpikeNeuronBase>
class Reinforcement : public ReinforcementBase {
    void serial_process() {
        begin() << "Constants: " << c << Self::end;
    }
    void linkWithNeuron(SpikeNeuronBase &_n) {
        try {
            Neuron &nc = static_cast<Neuron&>(_n);
            n.set(nc);
        } catch(std::bad_cast exp) {
            throw dnnException() << "Can't cast neuron for reinforcement " << name() << "\n";
        }
    }
protected:
    Ptr<Neuron> n;
    Constants c;
};


}
