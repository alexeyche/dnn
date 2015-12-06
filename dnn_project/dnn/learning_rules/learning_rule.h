#pragma once

#include <dnn/io/serialize.h>
#include <dnn/util/statistics.h>
#include <dnn/neurons/spike_neuron.h>
#include <dnn/util/ptr.h>
#include <dnn/util/interfaced_ptr.h>
#include <dnn/base/sim_element.h>

namespace dnn {


struct LearningRuleInterface {
	propSynSpikeDelegate propagateSynapseSpike;
	calculateDynamicsDelegate calculateDynamics;
	calculateDynamicsDelegate calculateDynamicsInternal;
};


class Network;
class Builder;

class LearningRuleBase : public SimElement {
friend class Network;
friend class Builder;
public:
	typedef LearningRuleInterface interface;

	static void __calculateDynamicsDefault(const Time &t) {}
	static void __propagateSynapseSpikeDefault(const SynSpike &s) {}

	static void provideDefaultInterface(LearningRuleInterface &i) {
		i.calculateDynamics = &LearningRuleBase::__calculateDynamicsDefault;
		i.propagateSynapseSpike =  &LearningRuleBase::__propagateSynapseSpikeDefault;
		i.calculateDynamicsInternal = &LearningRuleBase::__calculateDynamicsDefault;
	}

	template <typename T>
	void provideInterface(LearningRuleInterface &i) {
        i.calculateDynamics = MakeDelegate(static_cast<T*>(this), &T::calculateDynamics);
        i.propagateSynapseSpike = MakeDelegate(static_cast<T*>(this), &T::propagateSynapseSpike);
        i.calculateDynamicsInternal = MakeDelegate(static_cast<T*>(this), &T::calculateDynamicsInternal);
        ifc = i;
    }

	virtual void propagateSynapseSpike(const SynSpike &s) = 0;
	virtual void calculateDynamics(const Time &t) = 0;
	virtual void reset() = 0;

	Statistics& getStat() {
		return stat;
	}
	void calculateDynamicsInternal(const Time& t) {
		ifc.calculateDynamics(t);
	}



	virtual void linkWithNeuron(SpikeNeuronBase &_n) = 0;
protected:
	Statistics stat;
	LearningRuleInterface ifc;
};


template <typename Constants, typename State, typename Neuron>
class LearningRule : public LearningRuleBase {
public:
	void serial_process() {
		begin() << "Constants: " << c;
		if (messages->size() == 0) {
			(*this) << Self::end;
			return;
		}
		(*this) << "State: " << s << Self::end;

	}

	void linkWithNeuron(SpikeNeuronBase &_n) {
		try {
			Neuron &nc = static_cast<Neuron&>(_n);
			n.set(nc);
		} catch(std::bad_cast exp) {
			throw dnnException() << "Can't cast neuron for learning rule " << name() << "\n";
		}
	}

protected:
	Ptr<Neuron> n;

	State s;
	Constants c;
};




}

