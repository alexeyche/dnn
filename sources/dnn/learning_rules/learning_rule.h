#pragma once

#include <dnn/io/serialize.h>
#include <dnn/util/statistics.h>
#include <dnn/neurons/spike_neuron.h>
#include <dnn/util/ptr.h>
#include <dnn/util/interfaced_ptr.h>
#include <dnn/weight_normalizations/weight_normalization.h>

namespace dnn {


struct LearningRuleInterface {
	propSynSpikeDelegate propagateSynapseSpike;
	calculateDynamicsDelegate calculateDynamics;
};


class Network;
class Builder;

class LearningRuleBase : public SerializableBase {
friend class Network;
friend class Builder;
public:
	typedef LearningRuleInterface interface;

	static void __calculateDynamicsDefault(const Time &t) {}
	static void __propagateSynapseSpikeDefault(const SynSpike &s) {}

	static void provideDefaultInterface(LearningRuleInterface &i) {
		i.calculateDynamics = &LearningRuleBase::__calculateDynamicsDefault;
		i.propagateSynapseSpike =  &LearningRuleBase::__propagateSynapseSpikeDefault;
	}
	
	template <typename T>
	void provideInterface(LearningRuleInterface &i) {
        i.calculateDynamics = MakeDelegate(static_cast<T*>(this), &T::calculateDynamics);
        i.propagateSynapseSpike = MakeDelegate(static_cast<T*>(this), &T::propagateSynapseSpike);
    }

	virtual void propagateSynapseSpike(const SynSpike &s) = 0;
	virtual void calculateDynamics(const Time &t) = 0;
	virtual void reset() = 0;
		
	Statistics& getStat() {
		return stat;
	}
	void setWeightNormalization(WeightNormalizationBase *_norm) {
		norm.set(_norm);
	}

	virtual void linkWithNeuron(SpikeNeuronBase &_n) = 0;
protected:
	Statistics stat;
	InterfacedPtr<WeightNormalizationBase> norm;
};

/*@GENERATE_PROTO@*/
struct LearningRuleInfo : public Serializable<Protos::LearningRuleInfo> {
	void serial_process() {
		begin() << "weight_normalization_is_set: " << weight_normalization_is_set << Self::end;
	}
	bool weight_normalization_is_set;
};

template <typename Constants, typename State, typename Neuron>
class LearningRule : public LearningRuleBase {
public:	
	LearningRuleInfo getInfo() {
		LearningRuleInfo info;
		info.weight_normalization_is_set = norm.isSet();
		return info;
	}
	void serial_process() {
		begin() << "Constants: " << c;
		if (messages->size() == 0) {
			(*this) << Self::end;
			return;
		}
		(*this) << "State: " << s << Self::end;
		LearningRuleInfo info;
		if (mode == ProcessingOutput) {
			info = getInfo();
		}

		(*this) << "LearningRuleInfo: "   << info;
		if(info.weight_normalization_is_set) {
			(*this) << "WeightNormalization: " << norm;
		}
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

